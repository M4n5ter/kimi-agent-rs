use anyhow::{Result, anyhow};
use kaos::KaosPath;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::KaosConfig;
use crate::wire::WireMessageRecord;
use kosong::message::Message;

#[derive(Clone, Debug, PartialEq)]
pub struct KaosScopeRecord {
    pub id: String,
    pub kind: KaosScopeKind,
    pub display_name: String,
    pub definition_json: String,
    pub created_at: f64,
    pub updated_at: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KaosScopeKind {
    Local,
    Ssh,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceRecord {
    pub id: i64,
    pub kaos_scope_id: String,
    pub canonical_path: String,
    pub display_path: String,
    pub last_active_session_id: Option<i64>,
    pub created_at: f64,
    pub updated_at: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionOrigin {
    User,
    Subagent {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_tool_call_id: Option<String>,
        subagent_name: String,
    },
    System {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Empty,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionRecord {
    pub db_id: i64,
    pub id: String,
    pub workspace_id: i64,
    pub kaos_scope_id: String,
    pub work_dir: KaosPath,
    pub parent_session_id: Option<String>,
    pub root_session_id: String,
    pub origin: SessionOrigin,
    pub title: String,
    pub state: SessionState,
    pub token_count: i64,
    pub is_empty: bool,
    pub created_at: f64,
    pub updated_at: f64,
    pub last_activity_at: f64,
}

#[derive(Clone, Debug)]
pub struct CreateSession {
    pub work_dir: KaosPath,
    pub kaos: KaosConfig,
    pub session_id: Option<String>,
    pub parent_session_id: Option<String>,
    pub origin: SessionOrigin,
    pub title: Option<String>,
    pub state: SessionState,
}

#[derive(Clone, Debug)]
pub struct FinishSession {
    pub session_db_id: i64,
    pub state: SessionState,
    pub is_empty: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextEventRecord {
    pub seq: i64,
    pub created_at: f64,
    pub event: ContextEventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextMessageOrigin {
    UserInput,
    Synthetic,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ContextEventKind {
    Message {
        message: Message,
        origin: ContextMessageOrigin,
    },
    Usage {
        token_count: i64,
    },
    Checkpoint {
        checkpoint_id: i64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct WireEventRecord {
    pub seq: i64,
    pub created_at: f64,
    pub record: WireMessageRecord,
}

#[derive(Clone, Debug, PartialEq)]
pub struct McpServerRecord {
    pub name: String,
    pub transport_kind: String,
    pub config: Value,
}

impl KaosScopeKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Ssh => "ssh",
        }
    }
}

impl SessionState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Empty => "empty",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            "empty" => Ok(Self::Empty),
            _ => Err(anyhow!("unknown session state: {value}")),
        }
    }
}

impl SessionOrigin {
    pub(crate) fn kind_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Subagent { .. } => "subagent",
            Self::System { .. } => "system",
        }
    }
}

impl ContextMessageOrigin {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::UserInput => "user_input",
            Self::Synthetic => "synthetic",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "user_input" => Ok(Self::UserInput),
            "synthetic" => Ok(Self::Synthetic),
            _ => Err(anyhow!("unknown context message origin: {value}")),
        }
    }
}
