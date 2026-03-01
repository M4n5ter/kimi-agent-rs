use serde::{Deserialize, Serialize};

use crate::wire::{WireError, WireMessage, WireMessageEnvelope};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireMessageRecord {
    pub timestamp: f64,
    pub message: WireMessageEnvelope,
}

impl WireMessageRecord {
    pub fn from_wire_message(msg: &WireMessage, timestamp: f64) -> Result<Self, WireError> {
        Ok(Self {
            timestamp,
            message: WireMessageEnvelope::from_wire_message(msg)?,
        })
    }

    pub fn to_wire_message(&self) -> Result<WireMessage, WireError> {
        self.message.to_wire_message()
    }
}
