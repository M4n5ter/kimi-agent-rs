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

        let (shell_name, shell_path) = detect_unix_shell().await;

        Environment {
            os_kind,
            os_arch,
            os_version,
            shell_name,
            shell_path,
        }
    }
}

async fn detect_unix_shell() -> (String, KaosPath) {
    if let Some(shell) = shell_from_backend_env().await {
        return shell;
    }

    for (name, candidate) in unix_shell_candidates() {
        let path = KaosPath::new(candidate);
        if path.is_file(true).await {
            return (name.to_string(), path);
        }
    }

    ("sh".to_string(), KaosPath::new("/bin/sh"))
}

async fn shell_from_backend_env() -> Option<(String, KaosPath)> {
    let shell = kaos::env_var("SHELL").await.ok().flatten()?;
    if shell.is_empty() {
        return None;
    }

    let path = KaosPath::new(shell);
    let basename = path.name().to_ascii_lowercase();
    // Respect explicit bash/zsh shells from the backend environment, but do not let
    // `SHELL=/bin/sh` override our stronger bash/zsh fallback preference.
    let name = match basename.as_str() {
        "bash" => "bash",
        "zsh" => "zsh",
        _ => return None,
    };
    if !path.is_file(true).await {
        return None;
    }

    Some((name.to_string(), path))
}

fn unix_shell_candidates() -> [(&'static str, &'static str); 6] {
    [
        ("bash", "/bin/bash"),
        ("bash", "/usr/bin/bash"),
        ("bash", "/usr/local/bin/bash"),
        ("zsh", "/bin/zsh"),
        ("zsh", "/usr/bin/zsh"),
        ("zsh", "/usr/local/bin/zsh"),
    ]
}
