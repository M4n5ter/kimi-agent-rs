use std::collections::HashMap;
use std::sync::Arc;

use kimi_agent::soul::agent::AgentDefinition;
use serde_json::Value;

pub fn test_agent_definition() -> Arc<AgentDefinition> {
    let mocker_definition = Arc::new(AgentDefinition {
        name: "Mocker".to_string(),
        system_prompt: "You are a mock agent for testing.".to_string(),
        tool_paths: vec!["kimi_cli.tools.think:Think".to_string()],
        mcp_configs: Vec::<Value>::new(),
        fixed_subagents: Default::default(),
        fixed_subagent_descs: Default::default(),
    });

    Arc::new(AgentDefinition {
        name: "Root".to_string(),
        system_prompt: "You are the root agent for testing.".to_string(),
        tool_paths: vec![
            "kimi_cli.tools.multiagent:Task".to_string(),
            "kimi_cli.tools.multiagent:CreateSubagent".to_string(),
            "kimi_cli.tools.think:Think".to_string(),
        ],
        mcp_configs: Vec::<Value>::new(),
        fixed_subagents: HashMap::from([("mocker".to_string(), mocker_definition)]),
        fixed_subagent_descs: HashMap::from([(
            "mocker".to_string(),
            "The mock agent for testing purposes.".to_string(),
        )]),
    })
}
