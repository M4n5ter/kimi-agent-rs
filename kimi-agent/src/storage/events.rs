use anyhow::{Context, Result, anyhow};
use kosong::message::{Message, Role};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

use super::db::now_epoch_secs;
use super::{ContextEventKind, ContextEventRecord, Storage, WireEventRecord};
use crate::wire::{WireMessage, WireMessageRecord};

impl Storage {
    pub async fn append_context_messages(
        &self,
        session_db_id: i64,
        messages: &[Message],
    ) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        let messages = messages.to_vec();
        let now = now_epoch_secs();
        self.with_connection(move |conn| {
            let tx = conn.transaction()?;
            let mut next_seq = load_next_seq(&tx, session_db_id, "context")?;

            for message in &messages {
                let payload_json =
                    serde_json::to_string(message).context("serialize context message")?;
                tx.execute(
                    "
                    INSERT INTO session_events (session_id, stream, seq, created_at, kind, role, payload_json)
                    VALUES (?1, 'context', ?2, ?3, 'message', ?4, ?5)
                    ",
                    params![
                        session_db_id,
                        next_seq,
                        now,
                        role_label(&message.role),
                        payload_json,
                    ],
                )?;
                next_seq += 1;
            }

            update_session_stream_after_append(&tx, session_db_id, "context", next_seq, now)?;
            maybe_update_session_title(&tx, session_db_id, &messages)?;
            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn append_context_usage(&self, session_db_id: i64, token_count: i64) -> Result<()> {
        let now = now_epoch_secs();
        self.with_connection(move |conn| {
            let tx = conn.transaction()?;
            let next_seq = load_next_seq(&tx, session_db_id, "context")?;
            let payload_json = serde_json::json!({ "token_count": token_count }).to_string();
            tx.execute(
                "
                INSERT INTO session_events (session_id, stream, seq, created_at, kind, role, payload_json)
                VALUES (?1, 'context', ?2, ?3, 'usage', NULL, ?4)
                ",
                params![session_db_id, next_seq, now, payload_json],
            )?;
            update_session_stream_after_append(&tx, session_db_id, "context", next_seq + 1, now)?;
            tx.execute(
                "
                UPDATE sessions
                SET token_count = ?2, updated_at = ?3, last_activity_at = ?3, is_empty = 0
                WHERE id = ?1
                ",
                params![session_db_id, token_count, now],
            )?;
            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn append_context_checkpoint(
        &self,
        session_db_id: i64,
        checkpoint_id: i64,
    ) -> Result<()> {
        let now = now_epoch_secs();
        self.with_connection(move |conn| {
            let tx = conn.transaction()?;
            let next_seq = load_next_seq(&tx, session_db_id, "context")?;
            let payload_json = serde_json::json!({ "checkpoint_id": checkpoint_id }).to_string();
            tx.execute(
                "
                INSERT INTO session_events (session_id, stream, seq, created_at, kind, role, payload_json)
                VALUES (?1, 'context', ?2, ?3, 'checkpoint', NULL, ?4)
                ",
                params![session_db_id, next_seq, now, payload_json],
            )?;
            update_session_stream_after_append(&tx, session_db_id, "context", next_seq + 1, now)?;
            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn load_context_events(&self, session_db_id: i64) -> Result<Vec<ContextEventRecord>> {
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare(
                "
                SELECT seq, created_at, kind, payload_json
                FROM session_events
                WHERE session_id = ?1 AND stream = 'context'
                ORDER BY seq ASC
                ",
            )?;
            let rows = stmt.query_map(params![session_db_id], |row| {
                let seq: i64 = row.get(0)?;
                let created_at: f64 = row.get(1)?;
                let kind: String = row.get(2)?;
                let payload_json: String = row.get(3)?;
                let event = match kind.as_str() {
                    "message" => {
                        let message =
                            serde_json::from_str::<Message>(&payload_json).map_err(|err| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    3,
                                    rusqlite::types::Type::Text,
                                    Box::new(err),
                                )
                            })?;
                        ContextEventKind::Message(message)
                    }
                    "usage" => {
                        let payload =
                            serde_json::from_str::<Value>(&payload_json).map_err(|err| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    3,
                                    rusqlite::types::Type::Text,
                                    Box::new(err),
                                )
                            })?;
                        let token_count = payload
                            .get("token_count")
                            .and_then(Value::as_i64)
                            .ok_or_else(|| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    3,
                                    rusqlite::types::Type::Text,
                                    Box::new(std::io::Error::new(
                                        std::io::ErrorKind::InvalidData,
                                        "context usage missing token_count",
                                    )),
                                )
                            })?;
                        ContextEventKind::Usage { token_count }
                    }
                    "checkpoint" => {
                        let payload =
                            serde_json::from_str::<Value>(&payload_json).map_err(|err| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    3,
                                    rusqlite::types::Type::Text,
                                    Box::new(err),
                                )
                            })?;
                        let checkpoint_id = payload
                            .get("checkpoint_id")
                            .and_then(Value::as_i64)
                            .ok_or_else(|| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    3,
                                    rusqlite::types::Type::Text,
                                    Box::new(std::io::Error::new(
                                        std::io::ErrorKind::InvalidData,
                                        "context checkpoint missing checkpoint_id",
                                    )),
                                )
                            })?;
                        ContextEventKind::Checkpoint { checkpoint_id }
                    }
                    _ => {
                        return Err(rusqlite::Error::FromSqlConversionFailure(
                            2,
                            rusqlite::types::Type::Text,
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("unknown context event kind: {kind}"),
                            )),
                        ));
                    }
                };
                Ok(ContextEventRecord {
                    seq,
                    created_at,
                    event,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(Into::into)
        })
        .await
    }

    pub async fn truncate_context_from_seq(&self, session_db_id: i64, from_seq: i64) -> Result<()> {
        let now = now_epoch_secs();
        self.with_connection(move |conn| {
            let tx = conn.transaction()?;
            tx.execute(
                "
                DELETE FROM session_events
                WHERE session_id = ?1 AND stream = 'context' AND seq >= ?2
                ",
                params![session_db_id, from_seq],
            )?;
            refresh_context_session_state(&tx, session_db_id, now)?;
            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn clear_context_events(&self, session_db_id: i64) -> Result<()> {
        let now = now_epoch_secs();
        self.with_connection(move |conn| {
            let tx = conn.transaction()?;
            tx.execute(
                "
                DELETE FROM session_events
                WHERE session_id = ?1 AND stream = 'context'
                ",
                params![session_db_id],
            )?;
            refresh_context_session_state(&tx, session_db_id, now)?;
            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn append_wire_message(
        &self,
        session_db_id: i64,
        message: &WireMessage,
        timestamp: f64,
    ) -> Result<()> {
        let message = message.clone();
        let now = now_epoch_secs();
        self.with_connection(move |conn| {
            let tx = conn.transaction()?;
            let next_seq = load_next_seq(&tx, session_db_id, "wire")?;
            let record =
                WireMessageRecord::from_wire_message(&message, timestamp).map_err(anyhow::Error::msg)?;
            let payload_json = serde_json::to_string(&record).context("serialize wire message")?;
            tx.execute(
                "
                INSERT INTO session_events (session_id, stream, seq, created_at, kind, role, payload_json)
                VALUES (?1, 'wire', ?2, ?3, 'wire_message', NULL, ?4)
                ",
                params![session_db_id, next_seq, now, payload_json],
            )?;
            update_session_stream_after_append(&tx, session_db_id, "wire", next_seq + 1, now)?;
            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn load_wire_events(&self, session_db_id: i64) -> Result<Vec<WireEventRecord>> {
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare(
                "
                SELECT seq, created_at, payload_json
                FROM session_events
                WHERE session_id = ?1 AND stream = 'wire'
                ORDER BY seq ASC
                ",
            )?;
            let rows = stmt.query_map(params![session_db_id], |row| {
                let seq: i64 = row.get(0)?;
                let created_at: f64 = row.get(1)?;
                let payload_json: String = row.get(2)?;
                let record =
                    serde_json::from_str::<WireMessageRecord>(&payload_json).map_err(|err| {
                        rusqlite::Error::FromSqlConversionFailure(
                            2,
                            rusqlite::types::Type::Text,
                            Box::new(err),
                        )
                    })?;
                Ok(WireEventRecord {
                    seq,
                    created_at,
                    record,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(Into::into)
        })
        .await
    }
}

fn load_next_seq(conn: &Connection, session_db_id: i64, stream: &str) -> Result<i64> {
    let sql = match stream {
        "context" => "SELECT next_context_seq FROM sessions WHERE id = ?1",
        "wire" => "SELECT next_wire_seq FROM sessions WHERE id = ?1",
        _ => return Err(anyhow!("unknown session event stream: {stream}")),
    };
    conn.query_row(sql, params![session_db_id], |row| row.get::<_, i64>(0))
        .optional()?
        .ok_or_else(|| anyhow!("session not found: {session_db_id}"))
}

fn update_session_stream_after_append(
    conn: &Connection,
    session_db_id: i64,
    stream: &str,
    next_seq: i64,
    now: f64,
) -> Result<()> {
    let sql = match stream {
        "context" => {
            "
            UPDATE sessions
            SET next_context_seq = ?2, updated_at = ?3, last_activity_at = ?3, is_empty = 0
            WHERE id = ?1
            "
        }
        "wire" => {
            "
            UPDATE sessions
            SET next_wire_seq = ?2, updated_at = ?3, last_activity_at = ?3, is_empty = 0
            WHERE id = ?1
            "
        }
        _ => return Err(anyhow!("unknown session event stream: {stream}")),
    };
    conn.execute(sql, params![session_db_id, next_seq, now])?;
    Ok(())
}

fn refresh_context_session_state(conn: &Connection, session_db_id: i64, now: f64) -> Result<()> {
    let next_context_seq = conn.query_row(
        "
        SELECT COALESCE(MAX(seq) + 1, 0)
        FROM session_events
        WHERE session_id = ?1 AND stream = 'context'
        ",
        params![session_db_id],
        |row| row.get::<_, i64>(0),
    )?;
    let token_count = conn
        .query_row(
            "
        SELECT payload_json
        FROM session_events
        WHERE session_id = ?1 AND stream = 'context' AND kind = 'usage'
        ORDER BY seq DESC
        LIMIT 1
        ",
            params![session_db_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .and_then(|payload| {
            serde_json::from_str::<Value>(&payload)
                .ok()
                .and_then(|value| value.get("token_count").and_then(Value::as_i64))
        })
        .unwrap_or(0);
    let has_events = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM session_events WHERE session_id = ?1)",
        params![session_db_id],
        |row| row.get::<_, i64>(0),
    )?;
    conn.execute(
        "
        UPDATE sessions
        SET next_context_seq = ?2,
            token_count = ?3,
            is_empty = ?4,
            updated_at = ?5,
            last_activity_at = ?5
        WHERE id = ?1
        ",
        params![
            session_db_id,
            next_context_seq,
            token_count,
            i64::from(has_events == 0),
            now
        ],
    )?;
    Ok(())
}

fn maybe_update_session_title(
    conn: &Connection,
    session_db_id: i64,
    messages: &[Message],
) -> Result<()> {
    let Some(title_source) = messages.iter().find_map(session_title_from_message) else {
        return Ok(());
    };
    let session_id = conn.query_row(
        "SELECT session_id FROM sessions WHERE id = ?1",
        params![session_db_id],
        |row| row.get::<_, String>(0),
    )?;
    conn.execute(
        "
        UPDATE sessions
        SET title = ?2
        WHERE id = ?1 AND title = ?3
        ",
        params![
            session_db_id,
            format!("{} ({session_id})", shorten_text(&title_source, 50)),
            format!("Untitled ({session_id})"),
        ],
    )?;
    Ok(())
}

fn session_title_from_message(message: &Message) -> Option<String> {
    if message.role != Role::User {
        return None;
    }
    let text = message.extract_text(" ");
    let title = shorten_text(&text, 50);
    if title.is_empty() { None } else { Some(title) }
}

fn role_label(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn shorten_text(text: &str, width: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return String::new();
    }
    if collapsed.len() <= width {
        return collapsed;
    }
    let placeholder = "...";
    if width <= placeholder.len() {
        return placeholder.to_string();
    }
    let target = width - placeholder.len();
    let mut last_space = None;
    for (idx, ch) in collapsed.char_indices() {
        if idx > target {
            break;
        }
        if ch.is_whitespace() {
            last_space = Some(idx);
        }
    }
    let cut = last_space.unwrap_or(0);
    if cut == 0 {
        return placeholder.to_string();
    }
    format!("{}{}", &collapsed[..cut], placeholder)
}
