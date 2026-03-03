use std::sync::Arc;

use kimi_agent::soul::agent::{AgentDefinition, Runtime};
use serde_json::Value;

#[allow(dead_code)]
pub fn test_mocker_definition() -> Arc<AgentDefinition> {
    Arc::new(AgentDefinition {
        name: "Mocker".to_string(),
        system_prompt: "You are a mock agent for testing.".to_string(),
        tool_paths: vec!["kimi_cli.tools.think:Think".to_string()],
        mcp_configs: Vec::<Value>::new(),
    })
}

pub fn test_agent_definition() -> Arc<AgentDefinition> {
    Arc::new(AgentDefinition {
        name: "Root".to_string(),
        system_prompt: "You are the root agent for testing.".to_string(),
        tool_paths: vec![
            "kimi_cli.tools.multiagent:Task".to_string(),
            "kimi_cli.tools.multiagent:CreateSubagent".to_string(),
            "kimi_cli.tools.think:Think".to_string(),
        ],
        mcp_configs: Vec::<Value>::new(),
    })
}

#[allow(dead_code)]
pub async fn install_test_fixed_subagents(runtime: &Runtime) {
    runtime
        .subagent_registry
        .lock()
        .await
        .add_fixed_subagent(
            "mocker".to_string(),
            test_mocker_definition(),
            "The mock agent for testing purposes.".to_string(),
        )
        .expect("install fixed subagent");
}
