use std::collections::BTreeMap;

use anyhow::Result;
use kaos::{AsyncReadable, ExecOptions, Kaos, LocalKaos};
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
