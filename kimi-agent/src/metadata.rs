use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::debug;

use kaos::{Kaos, KaosPath, LocalKaos, get_current_kaos};

use crate::share::{ensure_share_dir, get_share_dir};

static METADATA_UPDATE_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

pub fn get_metadata_file() -> PathBuf {
    get_share_dir().join("kimi.json")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkDirMeta {
    pub path: String,
    #[serde(default = "default_kaos_name")]
    pub kaos: String,
    #[serde(default)]
    pub last_session_id: Option<String>,
}

impl WorkDirMeta {
    pub fn sessions_dir(&self) -> PathBuf {
        let hash = md5::compute(self.path.as_bytes());
        let hash_hex = format!("{:x}", hash);
        let dir_basename = if self.kaos == default_kaos_name() {
            hash_hex
        } else {
            format!("{}_{}", self.kaos, hash_hex)
        };
        get_share_dir().join("sessions").join(dir_basename)
    }

    pub async fn ensure_sessions_dir(&self) -> PathBuf {
        let dir = self.sessions_dir();
        tokio::fs::create_dir_all(&dir)
            .await
            .unwrap_or_else(|err| panic!("Failed to create sessions dir {}: {err}", dir.display()));
        dir
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(default)]
    pub work_dirs: Vec<WorkDirMeta>,
}

impl Metadata {
    pub fn get_work_dir_meta(&self, path: &KaosPath) -> Option<WorkDirMeta> {
        let kaos_name = get_current_kaos().storage_name();
        self.work_dirs
            .iter()
            .find(|wd| wd.path == path.to_string() && wd.kaos == kaos_name)
            .cloned()
    }

    pub fn new_work_dir_meta(&mut self, path: &KaosPath) -> WorkDirMeta {
        let meta = WorkDirMeta {
            path: path.to_string(),
            kaos: get_current_kaos().storage_name(),
            last_session_id: None,
        };
        self.work_dirs.push(meta.clone());
        meta
    }
}

pub async fn load_metadata() -> Metadata {
    let _ = ensure_share_dir().await;
    let metadata_file = get_metadata_file();
    debug!("Loading metadata from file: {}", metadata_file.display());
    if tokio::fs::metadata(&metadata_file).await.is_err() {
        debug!("No metadata file found, creating empty metadata");
        return Metadata::default();
    }
    let text = tokio::fs::read_to_string(&metadata_file)
        .await
        .unwrap_or_else(|err| {
            panic!(
                "Failed to read metadata file {}: {err}",
                metadata_file.display()
            )
        });
    serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("Invalid metadata file {}: {err}", metadata_file.display()))
}

pub async fn save_metadata(metadata: &Metadata) {
    let metadata_file = get_metadata_file();
    debug!("Saving metadata to file: {}", metadata_file.display());
    if let Some(parent) = metadata_file.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .unwrap_or_else(|err| {
                panic!("Failed to create metadata dir {}: {err}", parent.display())
            });
    }
    let text = serde_json::to_string_pretty(metadata).unwrap_or_else(|err| {
        panic!(
            "Failed to serialize metadata file {}: {err}",
            metadata_file.display()
        )
    });
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let temp_file =
        metadata_file.with_extension(format!("json.tmp-{}-{timestamp}", std::process::id()));
    tokio::fs::write(&temp_file, text)
        .await
        .unwrap_or_else(|err| {
            panic!(
                "Failed to write temporary metadata file {}: {err}",
                temp_file.display()
            )
        });
    tokio::fs::rename(&temp_file, &metadata_file)
        .await
        .unwrap_or_else(|err| {
            panic!(
                "Failed to replace metadata file {} with {}: {err}",
                metadata_file.display(),
                temp_file.display()
            )
        });
}

pub async fn update_metadata<R>(update: impl FnOnce(&mut Metadata) -> R) -> R {
    let _guard = METADATA_UPDATE_LOCK.lock().await;
    let mut metadata = load_metadata().await;
    let result = update(&mut metadata);
    save_metadata(&metadata).await;
    result
}

fn default_kaos_name() -> String {
    LocalKaos::new().storage_name()
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use tempfile::TempDir;
    use tokio::sync::{Barrier, Mutex};

    use super::{WorkDirMeta, load_metadata, update_metadata};

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::const_new(()));

    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: tests serialize env access with ENV_LOCK.
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev {
                // SAFETY: tests serialize env access with ENV_LOCK.
                unsafe {
                    std::env::set_var(self.key, prev);
                }
            } else {
                // SAFETY: tests serialize env access with ENV_LOCK.
                unsafe {
                    std::env::remove_var(self.key);
                }
            }
        }
    }

    #[tokio::test]
    async fn update_metadata_serializes_concurrent_writes() {
        let _lock = ENV_LOCK.lock().await;
        let share_dir = TempDir::new().expect("share dir");
        let _env = EnvGuard::set(
            "KIMI_SHARE_DIR",
            share_dir.path().to_str().expect("share dir path"),
        );

        let n_tasks = 24usize;
        let barrier = std::sync::Arc::new(Barrier::new(n_tasks));
        let mut tasks = Vec::new();
        for idx in 0..n_tasks {
            let barrier = std::sync::Arc::clone(&barrier);
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                update_metadata(move |metadata| {
                    metadata.work_dirs.push(WorkDirMeta {
                        path: format!("/tmp/work-{idx}"),
                        kaos: "local".to_string(),
                        last_session_id: Some(format!("session-{idx}")),
                    });
                })
                .await;
            }));
        }
        for task in tasks {
            task.await.expect("join metadata task");
        }

        let metadata = load_metadata().await;
        assert_eq!(metadata.work_dirs.len(), n_tasks);
    }
}
