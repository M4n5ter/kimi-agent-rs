use std::path::Path;

use tempfile::TempDir;
use tokio::sync::Mutex;

use kaos::KaosPath;
use kimi_agent::config::{KaosConfig, StorageConfig};
use kimi_agent::session::Session;
use kimi_agent::storage::{ContextMessageOrigin, SessionOrigin, SessionState, Storage};
use kosong::message::{Message, Role, TextPart};

static ENV_LOCK: Mutex<()> = Mutex::const_new(());

struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev {
            unsafe {
                std::env::set_var(self.key, prev);
            }
        } else {
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

async fn open_test_storage(root: &Path) -> Storage {
    Storage::open(&StorageConfig {
        database_path: root.join("state.db").display().to_string(),
        busy_timeout_ms: 1_000,
    })
    .await
    .expect("open test storage")
}

fn text_message(role: Role, text: &str) -> Message {
    Message::new(role, vec![TextPart::new(text).into()])
}

#[tokio::test]
async fn test_create_sets_fallback_title() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());

    let session = Session::create(storage, KaosConfig::Local, work_path, None)
        .await
        .expect("create session");
    assert!(session.title.starts_with("Untitled ("));
    assert!(session.is_empty().await.expect("session empty"));
}

#[tokio::test]
async fn test_find_uses_first_user_message_title() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());

    let session = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("title-session".to_string()),
    )
    .await
    .expect("create session");
    storage
        .append_context_messages(
            session.db_id(),
            &[text_message(
                Role::User,
                "hello world from sqlite session title",
            )],
            ContextMessageOrigin::UserInput,
        )
        .await
        .expect("append context message");

    let found = Session::find(storage, KaosConfig::Local, work_path, &session.id)
        .await
        .expect("find session")
        .expect("session");
    assert!(
        found
            .title
            .starts_with("hello world from sqlite session title")
    );
}

#[tokio::test]
async fn test_find_ignores_synthetic_user_messages_for_title() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());

    let session = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("synthetic-title-session".to_string()),
    )
    .await
    .expect("create session");
    storage
        .append_context_messages(
            session.db_id(),
            &[text_message(Role::User, "<system>CHECKPOINT 0</system>")],
            ContextMessageOrigin::Synthetic,
        )
        .await
        .expect("append synthetic checkpoint message");
    storage
        .append_context_messages(
            session.db_id(),
            &[text_message(
                Role::User,
                "real title from first user request",
            )],
            ContextMessageOrigin::UserInput,
        )
        .await
        .expect("append real user message");

    let found = Session::find(storage, KaosConfig::Local, work_path, &session.id)
        .await
        .expect("find session")
        .expect("session");
    assert!(
        found
            .title
            .starts_with("real title from first user request")
    );
}

#[tokio::test]
async fn test_find_uses_first_turn_text_when_context_has_no_user_input() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());

    let session = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("turn-title-session".to_string()),
    )
    .await
    .expect("create session");
    storage
        .maybe_update_session_title_from_turn_text(session.db_id(), "/init bootstrap the repo")
        .await
        .expect("update title from turn text");

    let found = Session::find(storage, KaosConfig::Local, work_path, &session.id)
        .await
        .expect("find session")
        .expect("session");
    assert!(found.title.starts_with("/init bootstrap the repo"));
}

#[tokio::test]
async fn test_list_sorts_by_updated_and_filters_empty_sessions() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());

    let empty = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("empty".to_string()),
    )
    .await
    .expect("create empty session");
    let first = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("first".to_string()),
    )
    .await
    .expect("create first session");
    storage
        .append_context_messages(
            first.db_id(),
            &[text_message(Role::User, "old session")],
            ContextMessageOrigin::UserInput,
        )
        .await
        .expect("append first message");
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let second = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("second".to_string()),
    )
    .await
    .expect("create second session");
    storage
        .append_context_messages(
            second.db_id(),
            &[text_message(Role::User, "new session")],
            ContextMessageOrigin::UserInput,
        )
        .await
        .expect("append second message");

    let sessions = Session::list(storage, KaosConfig::Local, work_path)
        .await
        .expect("list sessions");

    assert_eq!(sessions.len(), 2);
    assert_eq!(sessions[0].id, second.id);
    assert_eq!(sessions[1].id, first.id);
    assert!(sessions.iter().all(|session| session.id != empty.id));
}

#[tokio::test]
async fn test_continue_without_last_returns_none() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());

    let result = Session::continue_(storage, KaosConfig::Local, work_path)
        .await
        .expect("continue session");
    assert!(result.is_none());
}

#[tokio::test]
async fn test_continue_uses_last_active_root_session() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());

    let session = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("continued".to_string()),
    )
    .await
    .expect("create session");
    storage
        .append_context_messages(
            session.db_id(),
            &[text_message(Role::User, "persist me")],
            ContextMessageOrigin::UserInput,
        )
        .await
        .expect("append context");
    storage
        .finish_session(kimi_agent::storage::FinishSession {
            session_db_id: session.db_id(),
            state: SessionState::Completed,
            is_empty: false,
        })
        .await
        .expect("finish session");

    let resumed = Session::continue_(storage, KaosConfig::Local, work_path)
        .await
        .expect("continue session")
        .expect("resumed session");
    assert_eq!(resumed.id, session.id);
}

#[tokio::test]
async fn test_create_named_session() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());
    let session_id = "my-named-session".to_string();

    let session = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some(session_id.clone()),
    )
    .await
    .expect("create named session");

    assert_eq!(session.id, session_id);
    let found = Session::find(storage, KaosConfig::Local, work_path, &session.id)
        .await
        .expect("find session")
        .expect("session");
    assert_eq!(found.id, session.id);
}

#[tokio::test]
async fn test_list_excludes_child_sessions() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let work_dir = TempDir::new().expect("work dir");
    let work_path = KaosPath::from(work_dir.path().to_path_buf());

    let root = Session::create(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("root".to_string()),
    )
    .await
    .expect("create root");
    storage
        .append_context_messages(
            root.db_id(),
            &[text_message(Role::User, "root session")],
            ContextMessageOrigin::UserInput,
        )
        .await
        .expect("append root context");
    let child = Session::create_with_origin(
        storage.clone(),
        KaosConfig::Local,
        work_path.clone(),
        Some("child".to_string()),
        Some(root.id.clone()),
        SessionOrigin::Subagent {
            parent_tool_call_id: Some("tool-call".to_string()),
            subagent_name: "worker".to_string(),
        },
    )
    .await
    .expect("create child");
    storage
        .append_context_messages(
            child.db_id(),
            &[text_message(Role::User, "child session")],
            ContextMessageOrigin::UserInput,
        )
        .await
        .expect("append child context");

    let sessions = Session::list(storage, KaosConfig::Local, work_path)
        .await
        .expect("list sessions");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, root.id);
}

#[tokio::test]
async fn test_same_named_session_can_exist_in_different_workspaces() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let _env = set_home_env(home_dir.path());
    let storage = open_test_storage(home_dir.path()).await;

    let first_work_dir = TempDir::new().expect("first work dir");
    let first_work_path = KaosPath::from(first_work_dir.path().to_path_buf());
    let second_work_dir = TempDir::new().expect("second work dir");
    let second_work_path = KaosPath::from(second_work_dir.path().to_path_buf());

    let first = Session::create(
        storage.clone(),
        KaosConfig::Local,
        first_work_path.clone(),
        Some("shared-name".to_string()),
    )
    .await
    .expect("create first named session");
    let second = Session::create(
        storage.clone(),
        KaosConfig::Local,
        second_work_path.clone(),
        Some("shared-name".to_string()),
    )
    .await
    .expect("create second named session");

    assert_eq!(first.id, "shared-name");
    assert_eq!(second.id, "shared-name");
    assert_ne!(first.db_id(), second.db_id());

    let first_found = Session::find(
        storage.clone(),
        KaosConfig::Local,
        first_work_path,
        "shared-name",
    )
    .await
    .expect("find first session")
    .expect("first session");
    let second_found = Session::find(storage, KaosConfig::Local, second_work_path, "shared-name")
        .await
        .expect("find second session")
        .expect("second session");

    assert_eq!(first_found.db_id(), first.db_id());
    assert_eq!(second_found.db_id(), second.db_id());
}
