use anyhow::Result;
use kaos::KaosPath;

use crate::config::KaosConfig;
use crate::storage::{
    CreateSession, FinishSession, SessionOrigin, SessionRecord, SessionState, Storage,
};

#[derive(Clone, Debug)]
pub struct Session {
    storage: Storage,
    db_id: i64,
    pub id: String,
    pub work_dir: KaosPath,
    pub kaos: KaosConfig,
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

impl Session {
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn db_id(&self) -> i64 {
        self.db_id
    }

    pub fn kaos(&self) -> &KaosConfig {
        &self.kaos
    }

    pub async fn is_empty(&self) -> Result<bool> {
        self.storage.session_is_empty(self.db_id).await
    }

    pub async fn create(
        storage: Storage,
        kaos: KaosConfig,
        work_dir: KaosPath,
        session_id: Option<String>,
    ) -> Result<Self> {
        Self::create_with_origin(
            storage,
            kaos,
            work_dir,
            session_id,
            None,
            SessionOrigin::User,
        )
        .await
    }

    pub async fn create_with_origin(
        storage: Storage,
        kaos: KaosConfig,
        work_dir: KaosPath,
        session_id: Option<String>,
        parent_session_id: Option<String>,
        origin: SessionOrigin,
    ) -> Result<Self> {
        let record = storage
            .create_session(CreateSession {
                work_dir: work_dir.canonical(),
                kaos: kaos.clone(),
                session_id,
                parent_session_id,
                origin,
                title: None,
                state: SessionState::Pending,
            })
            .await?;
        Ok(Self::from_record(storage, kaos, record))
    }

    pub async fn find(
        storage: Storage,
        kaos: KaosConfig,
        work_dir: KaosPath,
        session_id: &str,
    ) -> Result<Option<Self>> {
        let record = storage
            .get_session(&work_dir.canonical(), &kaos, session_id)
            .await?;
        Ok(record.map(|record| Self::from_record(storage, kaos, record)))
    }

    pub async fn list(storage: Storage, kaos: KaosConfig, work_dir: KaosPath) -> Result<Vec<Self>> {
        let sessions = storage
            .list_sessions(&work_dir.canonical(), &kaos)
            .await?
            .into_iter()
            .filter(|record| record.parent_session_id.is_none() && !record.is_empty)
            .map(|record| Self::from_record(storage.clone(), kaos.clone(), record))
            .collect();
        Ok(sessions)
    }

    pub async fn continue_(
        storage: Storage,
        kaos: KaosConfig,
        work_dir: KaosPath,
    ) -> Result<Option<Self>> {
        let record = storage
            .continue_session(&work_dir.canonical(), &kaos)
            .await?;
        Ok(record.map(|record| Self::from_record(storage, kaos, record)))
    }

    fn from_record(storage: Storage, kaos: KaosConfig, record: SessionRecord) -> Self {
        Self {
            storage,
            db_id: record.db_id,
            id: record.id,
            work_dir: record.work_dir,
            kaos,
            parent_session_id: record.parent_session_id,
            root_session_id: record.root_session_id,
            origin: record.origin,
            title: record.title,
            state: record.state,
            token_count: record.token_count,
            is_empty: record.is_empty,
            created_at: record.created_at,
            updated_at: record.updated_at,
            last_activity_at: record.last_activity_at,
        }
    }
}

pub async fn post_run(session: &Session, state: SessionState) -> Result<()> {
    let is_empty = session.is_empty().await?;
    session
        .storage
        .finish_session(FinishSession {
            session_db_id: session.db_id,
            state,
            is_empty,
        })
        .await
}
