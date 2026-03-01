use anyhow::{Result, anyhow};
use tracing::{debug, error};

use kosong::message::{Message, Role};

use crate::soul::message::system;
use crate::storage::{ContextEventKind, Storage};

#[derive(Clone, Debug)]
pub struct Context {
    storage: Storage,
    session_id: String,
    history: Vec<Message>,
    token_count: i64,
    next_checkpoint_id: i64,
}

impl Context {
    pub fn new(storage: Storage, session_id: String) -> Self {
        Self {
            storage,
            session_id,
            history: Vec::new(),
            token_count: 0,
            next_checkpoint_id: 0,
        }
    }

    pub async fn restore(&mut self) -> Result<bool> {
        debug!("Restoring context for session {}", self.session_id);
        if !self.history.is_empty() {
            error!("The context storage is already modified");
            return Err(anyhow!("The context storage is already modified"));
        }

        let events = self.storage.load_context_events(&self.session_id).await?;
        if events.is_empty() {
            debug!("Empty context stream, skipping restoration");
            return Ok(false);
        }

        for event in events {
            match event.event {
                ContextEventKind::Message(message) => self.history.push(message),
                ContextEventKind::Usage { token_count } => self.token_count = token_count,
                ContextEventKind::Checkpoint { checkpoint_id } => {
                    self.next_checkpoint_id = checkpoint_id + 1;
                }
            }
        }
        Ok(true)
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    pub fn token_count(&self) -> i64 {
        self.token_count
    }

    pub fn n_checkpoints(&self) -> i64 {
        self.next_checkpoint_id
    }

    pub async fn checkpoint(&mut self, add_user_message: bool) -> Result<()> {
        let checkpoint_id = self.next_checkpoint_id;
        self.next_checkpoint_id += 1;
        debug!("Checkpointing, ID: {}", checkpoint_id);

        self.storage
            .append_context_checkpoint(&self.session_id, checkpoint_id)
            .await?;

        if add_user_message {
            let message = Message::new(
                Role::User,
                vec![system(&format!("CHECKPOINT {checkpoint_id}"))],
            );
            self.append_messages(message).await?;
        }

        Ok(())
    }

    pub async fn revert_to(&mut self, checkpoint_id: i64) -> Result<()> {
        debug!("Reverting checkpoint, ID: {}", checkpoint_id);
        let events = self.storage.load_context_events(&self.session_id).await?;
        let Some(checkpoint_seq) = events.iter().find_map(|event| match event.event {
            ContextEventKind::Checkpoint { checkpoint_id: id } if id == checkpoint_id => {
                Some(event.seq)
            }
            _ => None,
        }) else {
            error!("Checkpoint {} does not exist", checkpoint_id);
            return Err(anyhow!("Checkpoint {checkpoint_id} does not exist"));
        };

        self.storage
            .truncate_context_from_seq(&self.session_id, checkpoint_seq)
            .await?;
        self.reset_in_memory_state();
        self.restore().await?;
        Ok(())
    }

    pub async fn clear(&mut self) -> Result<()> {
        debug!("Clearing context");
        self.storage.clear_context_events(&self.session_id).await?;
        self.reset_in_memory_state();
        Ok(())
    }

    pub async fn append_messages<M>(&mut self, messages: M) -> Result<()>
    where
        M: Into<ContextMessages>,
    {
        let messages = match messages.into() {
            ContextMessages::One(message) => vec![message],
            ContextMessages::Many(messages) => messages,
        };
        debug!("Appending message(s) to context: {:?}", messages);
        self.storage
            .append_context_messages(&self.session_id, &messages)
            .await?;
        self.history.extend(messages);
        Ok(())
    }

    pub async fn update_token_count(&mut self, token_count: i64) -> Result<()> {
        debug!("Updating token count in context: {}", token_count);
        self.storage
            .append_context_usage(&self.session_id, token_count)
            .await?;
        self.token_count = token_count;
        Ok(())
    }

    fn reset_in_memory_state(&mut self) {
        self.history.clear();
        self.token_count = 0;
        self.next_checkpoint_id = 0;
    }
}

pub enum ContextMessages {
    One(Message),
    Many(Vec<Message>),
}

impl From<Message> for ContextMessages {
    fn from(value: Message) -> Self {
        Self::One(value)
    }
}

impl From<Vec<Message>> for ContextMessages {
    fn from(value: Vec<Message>) -> Self {
        Self::Many(value)
    }
}
