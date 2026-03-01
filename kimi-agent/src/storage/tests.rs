use std::path::PathBuf;

use kaos::KaosPath;
use tempfile::TempDir;

use super::{CreateSession, FinishSession, SessionOrigin, SessionState, Storage};
use crate::config::{KaosConfig, StorageConfig};

#[tokio::test]
async fn open_initializes_schema_and_workspace_records() {
    let temp_dir = TempDir::new().expect("temp dir");
    let config = StorageConfig {
        database_path: temp_dir.path().join("state.db").display().to_string(),
        busy_timeout_ms: 1_000,
    };
    let storage = Storage::open(&config).await.expect("open storage");

    let workspace = storage
        .resolve_workspace(
            &KaosPath::from(PathBuf::from("/tmp/example")),
            &KaosConfig::Local,
        )
        .await
        .expect("resolve workspace");
    let again = storage
        .resolve_workspace(
            &KaosPath::from(PathBuf::from("/tmp/example")),
            &KaosConfig::Local,
        )
        .await
        .expect("resolve workspace again");

    assert_eq!(workspace.id, again.id);
    assert!(storage.database_path().ends_with("state.db"));
}

#[tokio::test]
async fn create_root_and_child_sessions_in_same_workspace() {
    let temp_dir = TempDir::new().expect("temp dir");
    let config = StorageConfig {
        database_path: temp_dir.path().join("state.db").display().to_string(),
        busy_timeout_ms: 1_000,
    };
    let storage = Storage::open(&config).await.expect("open storage");
    let work_dir = KaosPath::from(PathBuf::from("/tmp/workspace"));

    let root = storage
        .create_session(CreateSession {
            work_dir: work_dir.clone(),
            kaos: KaosConfig::Local,
            session_id: Some("root-session".to_string()),
            parent_session_id: None,
            origin: SessionOrigin::User,
            title: Some("Root".to_string()),
            state: SessionState::Pending,
        })
        .await
        .expect("create root session");
    let child = storage
        .create_session(CreateSession {
            work_dir: work_dir.clone(),
            kaos: KaosConfig::Local,
            session_id: Some("child-session".to_string()),
            parent_session_id: Some(root.id.clone()),
            origin: SessionOrigin::Subagent {
                parent_tool_call_id: Some("tool-call-1".to_string()),
                subagent_name: "worker".to_string(),
            },
            title: Some("Child".to_string()),
            state: SessionState::Pending,
        })
        .await
        .expect("create child session");

    assert_eq!(root.root_session_id, root.id);
    assert_eq!(child.parent_session_id.as_deref(), Some(root.id.as_str()));
    assert_eq!(child.root_session_id, root.id);
}

#[tokio::test]
async fn continue_session_uses_last_active_root_session() {
    let temp_dir = TempDir::new().expect("temp dir");
    let config = StorageConfig {
        database_path: temp_dir.path().join("state.db").display().to_string(),
        busy_timeout_ms: 1_000,
    };
    let storage = Storage::open(&config).await.expect("open storage");
    let work_dir = KaosPath::from(PathBuf::from("/tmp/workspace"));

    let root = storage
        .create_session(CreateSession {
            work_dir: work_dir.clone(),
            kaos: KaosConfig::Local,
            session_id: Some("root-session".to_string()),
            parent_session_id: None,
            origin: SessionOrigin::User,
            title: Some("Root".to_string()),
            state: SessionState::Running,
        })
        .await
        .expect("create root session");
    storage
        .finish_session(FinishSession {
            session_db_id: root.db_id,
            state: SessionState::Completed,
            is_empty: false,
        })
        .await
        .expect("finish session");

    let resumed = storage
        .continue_session(&work_dir, &KaosConfig::Local)
        .await
        .expect("continue session")
        .expect("root session");
    assert_eq!(resumed.id, root.id);
}
