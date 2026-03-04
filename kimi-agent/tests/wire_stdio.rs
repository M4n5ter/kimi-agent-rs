use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tempfile::TempDir;
use tokio::process::Command;
use tokio::sync::Mutex;

use kaos::KaosPath;
use kimi_agent::config::{KaosConfig, StorageConfig};
use kimi_agent::session::Session;
use kimi_agent::storage::{ContextMessageOrigin, FinishSession, SessionState, Storage};
use kosong::message::{Message, Role, TextPart};

static ENV_LOCK: Mutex<()> = Mutex::const_new(());

struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        // SAFETY: tests serialize env access via ENV_LOCK to avoid races.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev {
            // SAFETY: tests serialize env access via ENV_LOCK to avoid races.
            unsafe {
                std::env::set_var(self.key, prev);
            }
        } else {
            // SAFETY: tests serialize env access via ENV_LOCK to avoid races.
            unsafe {
                std::env::remove_var(self.key);
            }
        }
    }
}

fn set_home_env(path: &Path) -> Vec<EnvGuard> {
    let share_dir = path.join(".kimi");
    vec![
        EnvGuard::set("HOME", path.to_str().expect("home path")),
        EnvGuard::set("USERPROFILE", path.to_str().expect("home path")),
        EnvGuard::set(
            "KIMI_SHARE_DIR",
            share_dir.to_str().expect("share dir path"),
        ),
    ]
}

fn storage_config(root: &Path) -> StorageConfig {
    StorageConfig {
        database_path: root.join(".kimi").join("state.db").display().to_string(),
        busy_timeout_ms: 1_000,
    }
}

async fn open_test_storage(root: &Path) -> Storage {
    Storage::open(&storage_config(root))
        .await
        .expect("open test storage")
}

fn text_message(role: Role, text: &str) -> Message {
    Message::new(role, vec![TextPart::new(text).into()])
}

fn kimi_agent_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_kimi-agent"))
}

async fn run_stdio_wire(session_id: &str, work_dir: &Path) -> std::process::Output {
    run_stdio_wire_with_args(&["--session", session_id], work_dir).await
}

async fn run_stdio_wire_with_args(args: &[&str], work_dir: &Path) -> std::process::Output {
    let mut child = Command::new(kimi_agent_bin())
        .args(args)
        .current_dir(work_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn kimi-agent");
    drop(child.stdin.take());
    tokio::time::timeout(Duration::from_secs(10), child.wait_with_output())
        .await
        .expect("wait for kimi-agent stdio exit")
        .expect("collect kimi-agent stdio output")
}

fn assert_success(output: &std::process::Output) {
    assert!(
        output.status.success(),
        "kimi-agent exited with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
async fn test_wire_stdio_exit_without_prompt_discards_new_session() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let work_dir = TempDir::new().expect("work dir");

    let output = run_stdio_wire("new-session", work_dir.path()).await;
    assert_success(&output);

    let storage = open_test_storage(home_dir.path()).await;
    let persisted = Session::find(
        storage,
        KaosConfig::Local,
        KaosPath::from(work_dir.path().to_path_buf()),
        "new-session",
    )
    .await
    .expect("find session");
    assert!(
        persisted.is_none(),
        "unused stdio session should be discarded"
    );
}

#[tokio::test]
async fn test_wire_stdio_startup_failure_discards_new_session() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let work_dir = TempDir::new().expect("work dir");
    let agent_file = work_dir.path().join("broken-agent.yaml");
    std::fs::write(
        &agent_file,
        r#"version: 1
agent:
  name: "Broken Agent"
  system_prompt_path: ./missing-system.md
  tools: ["kimi_cli.tools.think:Think"]
"#,
    )
    .expect("write broken agent");

    let output = run_stdio_wire_with_args(
        &[
            "--session",
            "failed-startup-session",
            "--agent-file",
            agent_file.to_str().expect("agent file path"),
        ],
        work_dir.path(),
    )
    .await;
    assert!(
        !output.status.success(),
        "kimi-agent unexpectedly succeeded\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let storage = open_test_storage(home_dir.path()).await;
    let persisted = Session::find(
        storage,
        KaosConfig::Local,
        KaosPath::from(work_dir.path().to_path_buf()),
        "failed-startup-session",
    )
    .await
    .expect("find session");
    assert!(
        persisted.is_none(),
        "startup-failed stdio session should be discarded"
    );
}

#[tokio::test]
async fn test_wire_stdio_exit_without_prompt_preserves_existing_session_state() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());
    let storage = open_test_storage(home_dir.path()).await;

    let session = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("existing-session".to_string()),
    )
    .await
    .expect("create session");
    storage
        .append_context_messages(
            session.db_id(),
            &[text_message(Role::User, "persisted content")],
            ContextMessageOrigin::UserInput,
        )
        .await
        .expect("append context message");
    storage
        .finish_session(FinishSession {
            session_db_id: session.db_id(),
            state: SessionState::Completed,
            is_empty: false,
        })
        .await
        .expect("finish session");

    let output = run_stdio_wire("existing-session", work_dir.path()).await;
    assert_success(&output);

    let persisted = Session::find(storage, KaosConfig::Local, work_path, "existing-session")
        .await
        .expect("find session")
        .expect("existing session");
    assert_eq!(persisted.state.as_str(), "completed");
    assert!(!persisted.is_empty().await.expect("session empty"));
}
