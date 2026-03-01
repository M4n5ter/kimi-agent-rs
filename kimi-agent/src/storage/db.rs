use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use rusqlite::Connection;

use super::{Storage, StorageInner, schema};
use crate::config::StorageConfig;

impl Storage {
    pub async fn open(config: &StorageConfig) -> Result<Self> {
        let database_path = expand_user(Path::new(&config.database_path));
        if let Some(parent) = database_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("create SQLite storage directory {}", parent.display()))?;
        }

        let storage = Self {
            inner: std::sync::Arc::new(StorageInner {
                database_path,
                busy_timeout: Duration::from_millis(config.busy_timeout_ms),
                operation_lock: tokio::sync::Mutex::new(()),
            }),
        };
        storage.initialize().await?;
        Ok(storage)
    }

    pub fn database_path(&self) -> &Path {
        &self.inner.database_path
    }

    async fn initialize(&self) -> Result<()> {
        self.with_connection(move |conn| {
            initialize_database(conn)?;
            schema::apply(conn).context("apply storage schema migrations")?;
            Ok(())
        })
        .await
    }

    pub(super) async fn with_connection<T, F>(&self, op: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    {
        let _guard = self.inner.operation_lock.lock().await;
        let database_path = self.inner.database_path.clone();
        let busy_timeout = self.inner.busy_timeout;
        tokio::task::spawn_blocking(move || {
            let mut conn = Connection::open(&database_path)
                .with_context(|| format!("open SQLite database {}", database_path.display()))?;
            configure_connection(&conn, busy_timeout)?;
            op(&mut conn)
        })
        .await
        .map_err(|err| anyhow!("storage worker join failed: {err}"))?
    }
}

pub(super) fn bool_to_i64(value: bool) -> i64 {
    i64::from(u8::from(value))
}

pub(super) fn now_epoch_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn configure_connection(conn: &Connection, busy_timeout: Duration) -> Result<()> {
    conn.busy_timeout(busy_timeout)?;
    conn.execute_batch(
        "
        PRAGMA foreign_keys = ON;
        PRAGMA temp_store = MEMORY;
        ",
    )?;
    Ok(())
}

fn initialize_database(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        ",
    )?;
    Ok(())
}

fn expand_user(path: &Path) -> PathBuf {
    let Some(home) = dirs::home_dir() else {
        return path.to_path_buf();
    };
    let path_str = path.to_string_lossy();
    if path_str == "~" {
        return home;
    }
    if let Some(stripped) = path_str.strip_prefix("~/") {
        return home.join(stripped);
    }
    path.to_path_buf()
}
