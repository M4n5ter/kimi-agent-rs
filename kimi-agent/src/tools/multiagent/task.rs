use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use kosong::message::Role;
use kosong::tooling::{CallableTool2, ToolReturnValue, tool_error, tool_ok};

use crate::session::post_run;
use crate::soul::agent::{AgentDefinition, Runtime};
use crate::soul::context::Context;
use crate::soul::kimisoul::KimiSoul;
use crate::soul::toolset::{KimiToolset, get_current_tool_call_or_none};
use crate::soul::{MaxStepsReached, get_current_wire_or_none, run_soul};
use crate::storage::{SessionOrigin, SessionState};
use crate::tools::utils::load_desc;
use crate::wire::{SubagentEvent, Wire, WireMessage, WireRecordTarget};

const TASK_DESC: &str = include_str!("../desc/multiagent/task.md");

const MAX_CONTINUE_ATTEMPTS: i64 = 1;

const CONTINUE_PROMPT: &str = "Your previous response was too brief. Please provide a more comprehensive summary that includes:\n\n1. Specific technical details and implementations\n2. Complete code examples if relevant\n3. Detailed findings and analysis\n4. All important information that should be aware of by the caller";

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TaskParams {
    #[schemars(description = "A short (3-5 word) description of the task")]
    pub description: String,
    #[schemars(description = "The name of the specialized subagent to use for this task")]
    pub subagent_name: String,
    #[schemars(
        description = "The task for the subagent to perform. You must provide a detailed prompt with all necessary background information because the subagent cannot see anything in your context."
    )]
    pub prompt: String,
}

pub struct TaskTool {
    description: String,
    toolset: Arc<tokio::sync::Mutex<KimiToolset>>,
    runtime: Runtime,
}

impl TaskTool {
    pub fn new(runtime: &Runtime, toolset: Arc<tokio::sync::Mutex<KimiToolset>>) -> Self {
        let subagents_md = {
            runtime
                .subagent_registry
                .try_lock()
                .map(|registry| registry.fixed_subagent_descriptions())
                .unwrap_or_default()
                .into_iter()
                .map(|(name, desc)| format!("- `{}`: {}", name, desc))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let desc = load_desc(TASK_DESC, &[("SUBAGENTS_MD", subagents_md)]);
        Self {
            description: desc,
            toolset,
            runtime: runtime.clone(),
        }
    }

    async fn run_subagent(
        &self,
        definition: Arc<AgentDefinition>,
        subagent_name: String,
        prompt: String,
    ) -> ToolReturnValue {
        let super_wire = match get_current_wire_or_none() {
            Some(wire) => wire,
            None => {
                return tool_error(
                    "",
                    "Wire is not available for subagent execution.",
                    "Wire unavailable",
                );
            }
        };
        let tool_call = match get_current_tool_call_or_none() {
            Some(call) => call,
            None => {
                return tool_error(
                    "",
                    "Tool call context is required for subagent execution.",
                    "Missing tool call",
                );
            }
        };
        let task_tool_call_id = tool_call.id.clone();
        let child_session = match crate::session::Session::create_with_origin(
            self.runtime.storage.clone(),
            self.runtime.config.kaos.clone(),
            self.runtime.session.work_dir.clone(),
            None,
            Some(self.runtime.session.id.clone()),
            SessionOrigin::Subagent {
                parent_tool_call_id: Some(task_tool_call_id.clone()),
                subagent_name,
            },
        )
        .await
        {
            Ok(session) => session,
            Err(err) => {
                return tool_error(
                    "",
                    format!("Failed to create subagent session: {err}"),
                    "Failed to run subagent",
                );
            }
        };
        let child_runtime = self.runtime.rebind_session(child_session.clone());
        let overlay = self.toolset.lock().await.snapshot_overlay();
        let agent = match definition
            .instantiate_with_overlay(child_runtime, &overlay)
            .await
        {
            Ok(agent) => agent,
            Err(err) => {
                let _ = child_session.delete().await;
                return tool_error(
                    "",
                    format!("Failed to initialize subagent: {err}"),
                    "Failed to run subagent",
                );
            }
        };

        let context = Context::new(
            agent.runtime.storage.clone(),
            child_session.db_id(),
            child_session.id.clone(),
        );
        let soul = std::sync::Arc::new(KimiSoul::new(agent, context));
        let (state, response) = self
            .execute_subagent_soul(
                std::sync::Arc::clone(&soul),
                std::sync::Arc::clone(&super_wire),
                task_tool_call_id,
                child_session.db_id(),
                prompt,
            )
            .await;
        soul.shutdown().await;
        let _ = post_run(&child_session, state).await;
        response
    }

    async fn execute_subagent_soul(
        &self,
        soul: Arc<KimiSoul>,
        super_wire: Arc<Wire>,
        task_tool_call_id: String,
        child_session_db_id: i64,
        prompt: String,
    ) -> (SessionState, ToolReturnValue) {
        let result = self
            .run_subagent_turn(
                Arc::clone(&soul),
                Arc::clone(&super_wire),
                task_tool_call_id.clone(),
                child_session_db_id,
                prompt,
            )
            .await;
        if let Err(err) = result {
            let response = if let Some(MaxStepsReached { n_steps }) =
                err.downcast_ref::<MaxStepsReached>()
            {
                tool_error(
                    "",
                    format!(
                        "Max steps {n_steps} reached when running subagent. Please try splitting the task into smaller subtasks."
                    ),
                    "Max steps reached",
                )
            } else {
                tool_error(
                    "",
                    format!("Failed to run subagent: {err}"),
                    "Failed to run subagent",
                )
            };
            return (SessionState::Failed, response);
        }

        let mut final_text = self.final_subagent_text(&soul).await;
        if final_text.is_empty() {
            return (
                SessionState::Failed,
                tool_error(
                    "",
                    "The subagent seemed not to run properly. Maybe you have to do the task yourself.",
                    "Failed to run subagent",
                ),
            );
        }

        if final_text.len() < 200 && MAX_CONTINUE_ATTEMPTS > 0 {
            let _ = self
                .run_subagent_turn(
                    Arc::clone(&soul),
                    super_wire,
                    task_tool_call_id,
                    child_session_db_id,
                    CONTINUE_PROMPT.to_string(),
                )
                .await;
            final_text = self.final_subagent_text(&soul).await;
        }

        if final_text.is_empty() {
            return (
                SessionState::Failed,
                tool_error(
                    "",
                    "The subagent seemed not to run properly. Maybe you have to do the task yourself.",
                    "Failed to run subagent",
                ),
            );
        }

        (SessionState::Completed, tool_ok(final_text, "", ""))
    }

    async fn run_subagent_turn(
        &self,
        soul: Arc<KimiSoul>,
        super_wire: Arc<Wire>,
        task_tool_call_id: String,
        child_session_db_id: i64,
        prompt: String,
    ) -> anyhow::Result<()> {
        let ui_loop = move |wire: Arc<Wire>| {
            let super_wire = Arc::clone(&super_wire);
            let task_tool_call_id = task_tool_call_id.clone();
            async move {
                let ui = wire.ui_side(true);
                loop {
                    let msg = ui.receive().await?;
                    match msg {
                        WireMessage::ApprovalRequest(_)
                        | WireMessage::ApprovalResponse(_)
                        | WireMessage::ToolCallRequest(_) => {
                            super_wire.soul_side().send(msg);
                        }
                        other => {
                            if let Ok(event) = SubagentEvent::new(task_tool_call_id.clone(), other)
                            {
                                super_wire
                                    .soul_side()
                                    .send(WireMessage::SubagentEvent(event));
                            }
                        }
                    }
                }
            }
        };
        let wire_target = Some(WireRecordTarget::new(
            self.runtime.storage.clone(),
            child_session_db_id,
        ));
        tokio::task::spawn_blocking(move || {
            let handle = tokio::runtime::Handle::current();
            handle.block_on(run_soul(
                soul.as_ref(),
                crate::wire::UserInput::from(prompt),
                ui_loop,
                CancellationToken::new(),
                wire_target,
            ))
        })
        .await
        .map_err(|err| anyhow::anyhow!("Failed to run subagent: {err}"))?
    }

    async fn final_subagent_text(&self, soul: &Arc<KimiSoul>) -> String {
        soul.context()
            .lock()
            .await
            .history()
            .last()
            .filter(|msg| msg.role == Role::Assistant)
            .map(|msg| msg.extract_text("\n"))
            .unwrap_or_default()
    }
}

#[async_trait::async_trait]
impl CallableTool2 for TaskTool {
    type Params = TaskParams;

    fn name(&self) -> &str {
        "Task"
    }

    fn description(&self) -> &str {
        &self.description
    }

    async fn call_typed(&self, params: Self::Params) -> ToolReturnValue {
        let agent = {
            let registry = self.runtime.subagent_registry.lock().await;
            match registry.get(&params.subagent_name) {
                Some(agent) => Arc::clone(&agent.definition),
                None => {
                    return tool_error(
                        "",
                        format!("Subagent not found: {}", params.subagent_name),
                        "Subagent not found",
                    );
                }
            }
        };

        self.run_subagent(agent, params.subagent_name, params.prompt)
            .await
    }
}
