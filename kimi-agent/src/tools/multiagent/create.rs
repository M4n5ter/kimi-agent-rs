use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;

use kosong::tooling::{CallableTool2, ToolReturnValue, tool_error, tool_ok};

use crate::soul::agent::{AgentDefinition, Runtime};
use crate::tools::utils::load_desc;

const CREATE_DESC: &str = include_str!("../desc/multiagent/create.md");

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateSubagentParams {
    #[schemars(
        description = "Unique name for this agent configuration (e.g., 'summarizer', 'code_reviewer'). This name will be used to reference the agent in the Task tool."
    )]
    pub name: String,
    #[schemars(
        description = "System prompt defining the agent's role, capabilities, and boundaries."
    )]
    pub system_prompt: String,
}

pub struct CreateSubagent {
    description: String,
    definition: Arc<AgentDefinition>,
    runtime: Runtime,
}

impl CreateSubagent {
    pub fn new(runtime: &Runtime, definition: Arc<AgentDefinition>) -> Self {
        Self {
            description: load_desc(CREATE_DESC, &[]),
            definition,
            runtime: runtime.clone(),
        }
    }
}

#[async_trait::async_trait]
impl CallableTool2 for CreateSubagent {
    type Params = CreateSubagentParams;

    fn name(&self) -> &str {
        "CreateSubagent"
    }

    fn description(&self) -> &str {
        &self.description
    }

    async fn call_typed(&self, params: Self::Params) -> ToolReturnValue {
        let mut market = self.runtime.labor_market.lock().await;
        if self.definition.fixed_subagents.contains_key(&params.name)
            || market.all_dynamic_subagents().contains_key(&params.name)
        {
            return tool_error(
                "",
                format!("Subagent with name '{}' already exists.", params.name),
                "Subagent already exists",
            );
        }

        let subagent = self
            .definition
            .derive_dynamic(params.name.clone(), params.system_prompt);
        market.add_dynamic_subagent(params.name.clone(), subagent);

        let mut names: Vec<String> = self.definition.fixed_subagents.keys().cloned().collect();
        names.extend(market.all_dynamic_subagents().keys().cloned());
        names.sort();
        names.dedup();
        let output = format!("Available subagents: {}", names.join(", "));

        tool_ok(
            output,
            format!("Subagent '{}' created successfully.", params.name),
            "",
        )
    }
}
