mod db;
mod events;
mod mcp;
mod schema;
mod scope;
mod sessions;
#[cfg(test)]
mod tests;
mod types;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub use types::{
    ContextEventKind, ContextEventRecord, ContextMessageOrigin, CreateSession, FinishSession,
    KaosScopeKind, KaosScopeRecord, McpServerRecord, SessionOrigin, SessionRecord, SessionState,
    WireEventRecord, WorkspaceRecord,
};

#[derive(Clone, Debug)]
pub struct Storage {
    pub(super) inner: Arc<StorageInner>,
}

#[derive(Debug)]
pub(super) struct StorageInner {
    pub(super) database_path: PathBuf,
    pub(super) busy_timeout: Duration,
    pub(super) operation_lock: tokio::sync::Mutex<()>,
}
