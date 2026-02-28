use kaos::{KaosPath, get_current_kaos};

#[derive(Clone, Debug)]
pub struct Environment {
    pub os_kind: String,
    pub os_arch: String,
    pub os_version: String,
    pub shell_name: String,
    pub shell_path: KaosPath,
}

impl Environment {
    pub async fn detect() -> Self {
        let platform = kaos::platform();
        let os_kind = match platform.os.as_str() {
            "macos" => "macOS",
            "windows" => "Windows",
            "linux" => "Linux",
            other => other,
        }
        .to_string();

        let os_arch = platform.arch;
        let os_version = if get_current_kaos().name() == "local" {
            sysinfo::System::long_os_version().unwrap_or_default()
        } else {
            String::new()
        };

        if os_kind == "Windows" {
            return Environment {
                os_kind,
                os_arch,
                os_version,
                shell_name: "Windows PowerShell".to_string(),
                shell_path: KaosPath::new("powershell.exe"),
            };
        }

        if let Some(shell) = shell_from_backend_env().await {
            return Environment {
                os_kind,
                os_arch,
                os_version,
                shell_name: shell.0,
                shell_path: shell.1,
            };
        }

        let mut shell_name = "sh".to_string();
        let mut shell_path = KaosPath::new("/bin/sh");
        for candidate in ["/bin/bash", "/usr/bin/bash", "/usr/local/bin/bash"] {
            let path = KaosPath::new(candidate);
            if path.is_file(true).await {
                shell_name = "bash".to_string();
                shell_path = path;
                break;
            }
        }

        Environment {
            os_kind,
            os_arch,
            os_version,
            shell_name,
            shell_path,
        }
    }
}

async fn shell_from_backend_env() -> Option<(String, KaosPath)> {
    let shell = kaos::env_var("SHELL").await.ok().flatten()?;
    if shell.is_empty() {
        return None;
    }

    let path = KaosPath::new(shell);
    let basename = path.name().to_ascii_lowercase();
    let name = match basename.as_str() {
        "bash" => "bash",
        "sh" => "sh",
        _ => return None,
    };
    if !path.is_file(true).await {
        return None;
    }

    Some((name.to_string(), path))
}
