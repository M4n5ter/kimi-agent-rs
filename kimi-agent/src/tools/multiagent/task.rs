use schemars::JsonSchema;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use kosong::message::Role;
use kosong::tooling::{CallableTool2, ToolReturnValue, tool_error, tool_ok};

use crate::session::post_run;
use crate::soul::agent::{Agent, LaborMarket, Runtime};
use crate::soul::context::Context;
use crate::soul::kimisoul::KimiSoul;
use crate::soul::toolset::get_current_tool_call_or_none;
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
    labor_market: std::sync::Arc<tokio::sync::Mutex<LaborMarket>>,
    session: crate::session::Session,
}

impl TaskTool {
    pub fn new(runtime: &Runtime) -> Self {
        let subagents_md = {
            let mut names: Vec<(String, String)> = Vec::new();
            if let Ok(market) = runtime.labor_market.try_lock() {
                names = market
                    .fixed_subagent_descs()
                    .iter()
                    .map(|(name, desc)| (name.clone(), desc.clone()))
                    .collect();
                names.sort_by(|a, b| a.0.cmp(&b.0));
            }
            names
                .into_iter()
                .map(|(name, desc)| format!("- `{}`: {}", name, desc))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let desc = load_desc(TASK_DESC, &[("SUBAGENTS_MD", subagents_md)]);
        Self {
            description: desc,
            labor_market: runtime.labor_market.clone(),
            session: runtime.session.clone(),
        }
    }

    async fn run_subagent(
        &self,
        mut agent: Agent,
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
            self.session.storage().clone(),
            self.session.kaos().clone(),
            self.session.work_dir.clone(),
            None,
            Some(self.session.id.clone()),
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
        agent.runtime.session = child_session.clone();
        agent.runtime.storage = self.session.storage().clone();
        agent.runtime.config.kaos = self.session.kaos().clone();

        let make_ui_loop = |super_wire: std::sync::Arc<Wire>, task_tool_call_id: String| {
            move |wire: std::sync::Arc<Wire>| {
                let super_wire = std::sync::Arc::clone(&super_wire);
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
                                if let Ok(event) =
                                    SubagentEvent::new(task_tool_call_id.clone(), other)
                                {
                                    super_wire
                                        .soul_side()
                                        .send(WireMessage::SubagentEvent(event));
                                }
                            }
                        }
                    }
                }
            }
        };

        let context = Context::new(agent.runtime.storage.clone(), child_session.id.clone());
        let soul = std::sync::Arc::new(KimiSoul::new(agent, context));
        let soul_run = std::sync::Arc::clone(&soul);
        let ui_loop = make_ui_loop(
            std::sync::Arc::clone(&super_wire),
            task_tool_call_id.clone(),
        );
        let wire_target = Some(WireRecordTarget::new(
            self.session.storage().clone(),
            child_session.id.clone(),
        ));
        let result = match tokio::task::spawn_blocking(move || {
            let handle = tokio::runtime::Handle::current();
            handle.block_on(run_soul(
                soul_run.as_ref(),
                crate::wire::UserInput::from(prompt),
                ui_loop,
                CancellationToken::new(),
                wire_target,
            ))
        })
        .await
        {
            Ok(result) => result,
            Err(err) => {
                let _ = post_run(&child_session, SessionState::Failed).await;
                return tool_error(
                    "",
                    format!("Failed to run subagent: {err}"),
                    "Failed to run subagent",
                );
            }
        };

        if let Err(err) = result {
            let _ = post_run(&child_session, SessionState::Failed).await;
            if let Some(MaxStepsReached { n_steps }) = err.downcast_ref::<MaxStepsReached>() {
                return tool_error(
                    "",
                    format!(
                        "Max steps {n_steps} reached when running subagent. Please try splitting the task into smaller subtasks."
                    ),
                    "Max steps reached",
                );
            }
            return tool_error(
                "",
                format!("Failed to run subagent: {err}"),
                "Failed to run subagent",
            );
        }

        let history = soul.context().lock().await.history().to_vec();
        let mut final_message = history.last().cloned();
        let mut final_text = final_message
            .as_ref()
            .filter(|msg| msg.role == Role::Assistant)
            .map(|msg| msg.extract_text("\n"))
            .unwrap_or_default();

        if final_text.is_empty() {
            let _ = post_run(&child_session, SessionState::Failed).await;
            return tool_error(
                "",
                "The subagent seemed not to run properly. Maybe you have to do the task yourself.",
                "Failed to run subagent",
            );
        }

        if final_text.len() < 200 && MAX_CONTINUE_ATTEMPTS > 0 {
            let soul_run = std::sync::Arc::clone(&soul);
            let ui_loop = make_ui_loop(
                std::sync::Arc::clone(&super_wire),
                task_tool_call_id.clone(),
            );
            let storage = self.session.storage().clone();
            let child_session_id = child_session.id.clone();
            let _ = tokio::task::spawn_blocking(move || {
                let handle = tokio::runtime::Handle::current();
                handle.block_on(run_soul(
                    soul_run.as_ref(),
                    crate::wire::UserInput::from(CONTINUE_PROMPT),
                    ui_loop,
                    CancellationToken::new(),
                    Some(WireRecordTarget::new(storage, child_session_id)),
                ))
            })
            .await;
            let history = soul.context().lock().await.history().to_vec();
            final_message = history.last().cloned();
            final_text = final_message
                .as_ref()
                .filter(|msg| msg.role == Role::Assistant)
                .map(|msg| msg.extract_text("\n"))
                .unwrap_or_default();
        }

        if final_text.is_empty() {
            let _ = post_run(&child_session, SessionState::Failed).await;
            return tool_error(
                "",
                "The subagent seemed not to run properly. Maybe you have to do the task yourself.",
                "Failed to run subagent",
            );
        }

        let _ = post_run(&child_session, SessionState::Completed).await;
        tool_ok(final_text, "", "")
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
        let subagents = self.labor_market.lock().await.all_subagents();
        let agent = match subagents.get(&params.subagent_name) {
            Some(agent) => agent.clone(),
            None => {
                return tool_error(
                    "",
                    format!("Subagent not found: {}", params.subagent_name),
                    "Subagent not found",
                );
            }
        };

        self.run_subagent(agent, params.subagent_name, params.prompt)
            .await
    }
}
