use anyhow::{Result, anyhow};
use rusqlite::Connection;

pub const CURRENT_SCHEMA_VERSION: i64 = 2;

const MIGRATION_1: &str = "
CREATE TABLE kaos_scopes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    display_name TEXT NOT NULL,
    -- Sanitized backend definition used for debugging and scope reconstruction.
    definition_json TEXT NOT NULL CHECK (json_valid(definition_json)),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
) STRICT;

CREATE TABLE workspaces (
    id INTEGER PRIMARY KEY,
    kaos_scope_id TEXT NOT NULL REFERENCES kaos_scopes(id) ON DELETE RESTRICT,
    canonical_path TEXT NOT NULL,
    display_path TEXT NOT NULL,
    -- Internal row id of the last resumable top-level session for this workspace and Kaos scope.
    last_active_session_id INTEGER NULL REFERENCES sessions(id) ON DELETE SET NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (kaos_scope_id, canonical_path)
) STRICT;

CREATE TABLE sessions (
    -- Internal stable row id used by foreign keys and runtime persistence.
    id INTEGER PRIMARY KEY,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    -- User-visible session id accepted by --session, unique only inside a workspace.
    session_id TEXT NOT NULL,
    parent_session_id INTEGER NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    -- Internal row id of the stable tree root. Root sessions backfill this to their own id before commit.
    root_session_id INTEGER NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    origin_kind TEXT NOT NULL,
    -- Origin payload keyed by origin_kind, for example subagent metadata.
    origin_json TEXT NOT NULL CHECK (json_valid(origin_json)),
    title TEXT NOT NULL,
    state TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    is_empty INTEGER NOT NULL DEFAULT 1,
    -- Next append sequence for the context event stream.
    next_context_seq INTEGER NOT NULL DEFAULT 0,
    -- Next append sequence for the wire event stream.
    next_wire_seq INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_activity_at REAL NOT NULL,
    UNIQUE (workspace_id, session_id)
) STRICT;

CREATE TABLE session_events (
    id INTEGER PRIMARY KEY,
    -- Internal session row id that owns this event stream entry.
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    -- Logical stream name: context or wire.
    stream TEXT NOT NULL,
    -- Monotonic sequence within (session_id, stream).
    seq INTEGER NOT NULL,
    created_at REAL NOT NULL,
    -- Event discriminator within the stream, such as message or checkpoint.
    kind TEXT NOT NULL,
    role TEXT NULL,
    -- Canonical JSON payload for the event kind.
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    UNIQUE (session_id, stream, seq)
) STRICT;

CREATE TABLE mcp_servers (
    kaos_scope_id TEXT NOT NULL REFERENCES kaos_scopes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    transport_kind TEXT NOT NULL,
    -- Canonical persisted MCP server config entry.
    config_json TEXT NOT NULL CHECK (json_valid(config_json)),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (kaos_scope_id, name)
) STRICT;

CREATE TABLE mcp_credentials (
    kaos_scope_id TEXT NOT NULL REFERENCES kaos_scopes(id) ON DELETE CASCADE,
    server_url TEXT NOT NULL,
    -- Serialized OAuth credential payload for the MCP resource server.
    credentials_json TEXT NOT NULL CHECK (json_valid(credentials_json)),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (kaos_scope_id, server_url)
) STRICT;

CREATE INDEX workspaces_by_scope_path
    ON workspaces (kaos_scope_id, canonical_path);

CREATE INDEX sessions_by_workspace_updated
    ON sessions (workspace_id, updated_at DESC);

CREATE INDEX sessions_by_workspace_session_id
    ON sessions (workspace_id, session_id);

CREATE INDEX sessions_by_parent
    ON sessions (parent_session_id, created_at ASC);

CREATE INDEX sessions_by_root
    ON sessions (root_session_id, created_at ASC);

CREATE INDEX sessions_by_state
    ON sessions (state, updated_at DESC);

CREATE INDEX session_events_by_session_stream_seq
    ON session_events (session_id, stream, seq);

CREATE INDEX session_events_by_session_created
    ON session_events (session_id, created_at ASC);

CREATE INDEX session_events_by_stream_kind
    ON session_events (stream, kind);
";

pub fn apply(conn: &mut Connection) -> Result<()> {
    let version = conn.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))?;
    if version > CURRENT_SCHEMA_VERSION {
        return Err(anyhow!(
            "SQLite schema version {} is newer than supported version {}",
            version,
            CURRENT_SCHEMA_VERSION
        ));
    }

    if version == 0 {
        let tx = conn.transaction()?;
        tx.execute_batch(MIGRATION_1)?;
        tx.pragma_update(None, "user_version", CURRENT_SCHEMA_VERSION)?;
        tx.commit()?;
        return Ok(());
    }

    if version < CURRENT_SCHEMA_VERSION {
        return Err(anyhow!(
            "SQLite schema version {} is unsupported by this build. Delete the existing database and let kimi-agent recreate it with schema version {}.",
            version,
            CURRENT_SCHEMA_VERSION
        ));
    }

    Ok(())
}
