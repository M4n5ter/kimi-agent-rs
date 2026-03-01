use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use kaos::{AsyncReadable, ExecOptions, Kaos, KaosPath, LocalKaos};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::const_new(());

struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        // SAFETY: tests serialize environment mutations via ENV_LOCK.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, prev }
    }

    fn remove(key: &'static str) -> Self {
        let prev = std::env::var(key).ok();
        // SAFETY: tests serialize environment mutations via ENV_LOCK.
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev {
            // SAFETY: tests serialize environment mutations via ENV_LOCK.
            unsafe {
                std::env::set_var(self.key, prev);
            }
        } else {
            // SAFETY: tests serialize environment mutations via ENV_LOCK.
            unsafe {
                std::env::remove_var(self.key);
            }
        }
    }
}

fn print_env_command(key: &str) -> Vec<String> {
    #[cfg(windows)]
    {
        vec![
            "powershell.exe".to_string(),
            "-Command".to_string(),
            format!("[Console]::Out.Write($env:{key})"),
        ]
    }

    #[cfg(not(windows))]
    {
        vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            format!("printf '%s' \"${key}\""),
        ]
    }
}

async fn read_all(stream: &mut dyn AsyncReadable) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    loop {
        let chunk = stream.read(1024).await?;
        if chunk.is_empty() {
            break;
        }
        output.extend_from_slice(&chunk);
    }
    Ok(output)
}

async fn run_command(args: &[String], options: ExecOptions) -> Result<String> {
    let kaos = LocalKaos::new();
    let mut process = kaos.exec(args, options).await?;
    let stdout = match process.take_stdout() {
        Some(mut stdout) => read_all(stdout.as_mut()).await?,
        None => read_all(process.stdout()).await?,
    };
    let exit_code = process.wait().await?;
    assert_eq!(exit_code, 0);
    Ok(String::from_utf8(stdout).expect("stdout utf-8"))
}

#[tokio::test]
async fn exec_options_adds_child_environment_variables() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("KAOS_EXEC_CHILD_ONLY");

    let args = print_env_command("KAOS_EXEC_CHILD_ONLY");
    let output = run_command(
        &args,
        ExecOptions {
            cwd: None,
            env_overrides: BTreeMap::from([(
                "KAOS_EXEC_CHILD_ONLY".to_string(),
                "child-value".to_string(),
            )]),
        },
    )
    .await
    .expect("run command");

    assert_eq!(output, "child-value");
}

#[tokio::test]
async fn exec_options_overrides_inherited_environment_values() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("KAOS_EXEC_OVERRIDE", "parent-value");

    let args = print_env_command("KAOS_EXEC_OVERRIDE");
    let output = run_command(
        &args,
        ExecOptions {
            cwd: None,
            env_overrides: BTreeMap::from([(
                "KAOS_EXEC_OVERRIDE".to_string(),
                "child-value".to_string(),
            )]),
        },
    )
    .await
    .expect("run command");

    assert_eq!(output, "child-value");
}

#[tokio::test]
async fn empty_exec_options_preserve_inherited_environment() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("KAOS_EXEC_INHERITED", "parent-value");

    let args = print_env_command("KAOS_EXEC_INHERITED");
    let output = run_command(&args, ExecOptions::default())
        .await
        .expect("run command");

    assert_eq!(output, "parent-value");
}

#[tokio::test]
async fn exec_options_override_child_working_directory() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let test_dir =
        std::env::temp_dir().join(format!("kaos-exec-cwd-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&test_dir).expect("create test dir");

    #[cfg(windows)]
    let args = vec![
        "powershell.exe".to_string(),
        "-Command".to_string(),
        "[Console]::Out.Write((Get-Location).Path)".to_string(),
    ];

    #[cfg(not(windows))]
    let args = vec!["/bin/sh".to_string(), "-c".to_string(), "pwd".to_string()];

    let output = run_command(
        &args,
        ExecOptions {
            cwd: Some(KaosPath::from(test_dir.clone())),
            env_overrides: BTreeMap::new(),
        },
    )
    .await
    .expect("run command");

    let normalized_output = output.trim_end_matches(['\r', '\n']);
    let expected = std::fs::canonicalize(&test_dir).expect("canonical expected dir");
    let actual = std::fs::canonicalize(normalized_output).expect("canonical actual dir");
    assert_eq!(actual, expected);

    std::fs::remove_dir_all(&test_dir).expect("cleanup test dir");
}

#[tokio::test]
async fn local_kaos_connect_tcp_returns_duplex_stream() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind listener");
    let addr = listener.local_addr().expect("listener addr");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut buf = [0u8; 4];
        socket.read_exact(&mut buf).await.expect("read ping");
        assert_eq!(&buf, b"ping");
        socket.write_all(b"pong").await.expect("write pong");
    });

    let kaos = LocalKaos::new();
    let mut stream = kaos
        .connect_tcp("127.0.0.1", addr.port())
        .await
        .expect("connect tcp");
    stream.write_all(b"ping").await.expect("write ping");
    stream.flush().await.expect("flush ping");
    let mut buf = [0u8; 4];
    stream.read_exact(&mut buf).await.expect("read pong");
    assert_eq!(&buf, b"pong");

    server.await.expect("join server");
}
