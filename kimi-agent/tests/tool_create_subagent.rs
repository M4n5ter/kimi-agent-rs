mod agent_test_utils;
mod tool_test_utils;

use kimi_agent::session::Session;
use kimi_agent::soul::agent::AgentDefinition;
use kimi_agent::tools::multiagent::{CreateSubagent, CreateSubagentParams, TaskParams, TaskTool};
use kosong::tooling::CallableTool2;
use serde_json::json;
use std::sync::Arc;

use tool_test_utils::RuntimeFixture;

#[tokio::test]
async fn test_create_subagent() {
    let fixture = RuntimeFixture::new();
    agent_test_utils::install_test_fixed_subagents(&fixture.runtime).await;
    let tool = CreateSubagent::new(&fixture.runtime, agent_test_utils::test_agent_definition());

    let result = tool
        .call_typed(CreateSubagentParams {
            name: "test_agent".to_string(),
            system_prompt: "You are a test agent.".to_string(),
        })
        .await;

    assert!(!result.is_error);
    assert_eq!(
        result.output,
        kosong::tooling::ToolOutput::Text("Available subagents: mocker, test_agent".to_string())
    );
    assert_eq!(
        result.message,
        "Subagent 'test_agent' created successfully."
    );
    let subagents = fixture
        .runtime
        .subagent_registry
        .lock()
        .await
        .all_dynamic_subagents();
    assert!(subagents.contains_key("test_agent"));
}

#[tokio::test]
async fn test_create_existing_subagent() {
    let fixture = RuntimeFixture::new();
    agent_test_utils::install_test_fixed_subagents(&fixture.runtime).await;
    let tool = CreateSubagent::new(&fixture.runtime, agent_test_utils::test_agent_definition());

    let _ = tool
        .call_typed(CreateSubagentParams {
            name: "existing_agent".to_string(),
            system_prompt: "You are an existing agent.".to_string(),
        })
        .await;

    let subagents = fixture
        .runtime
        .subagent_registry
        .lock()
        .await
        .all_dynamic_subagents();
    assert!(subagents.contains_key("existing_agent"));

    let result = tool
        .call_typed(CreateSubagentParams {
            name: "existing_agent".to_string(),
            system_prompt: "You are an existing agent.".to_string(),
        })
        .await;

    assert!(result.is_error);
    assert_eq!(
        result.message,
        "Subagent with name 'existing_agent' already exists."
    );
    assert_eq!(result.brief(), "Subagent already exists");
    let subagents = fixture
        .runtime
        .subagent_registry
        .lock()
        .await
        .all_dynamic_subagents();
    assert!(subagents.contains_key("existing_agent"));
}

#[tokio::test]
async fn test_created_subagent_instantiates_with_child_session_runtime() {
    let fixture = RuntimeFixture::new();
    agent_test_utils::install_test_fixed_subagents(&fixture.runtime).await;
    let root_definition = agent_test_utils::test_agent_definition();
    let tool = CreateSubagent::new(&fixture.runtime, root_definition.clone());

    let result = tool
        .call_typed(CreateSubagentParams {
            name: "nested_agent".to_string(),
            system_prompt: "You are a nested agent.".to_string(),
        })
        .await;
    assert!(!result.is_error);

    let definition = fixture
        .runtime
        .subagent_registry
        .lock()
        .await
        .all_dynamic_subagents()
        .remove("nested_agent")
        .expect("created subagent definition");
    assert!(
        fixture
            .runtime
            .subagent_registry
            .lock()
            .await
            .contains("mocker")
    );

    let child_session = Session::create(
        fixture.runtime.storage.clone(),
        fixture.runtime.config.kaos.clone(),
        fixture.runtime.session.work_dir.clone(),
        Some("child-session".to_string()),
    )
    .await
    .expect("create child session");
    let child_runtime = fixture.runtime.rebind_session(child_session.clone());

    let root_agent = root_definition
        .instantiate(fixture.runtime.clone())
        .await
        .expect("instantiate root agent");
    let nested_agent = definition
        .instantiate(child_runtime)
        .await
        .expect("instantiate nested agent");

    assert_eq!(nested_agent.runtime.session.id, child_session.id);
    assert!(!std::sync::Arc::ptr_eq(
        &nested_agent.toolset,
        &root_agent.toolset
    ));
}

#[tokio::test]
async fn test_create_subagent_rejects_global_fixed_name_collision() {
    let fixture = RuntimeFixture::new();
    fixture
        .runtime
        .subagent_registry
        .lock()
        .await
        .add_fixed_subagent(
            "coder".to_string(),
            Arc::new(AgentDefinition {
                name: "Coder".to_string(),
                system_prompt: "You are a coder.".to_string(),
                tool_paths: vec!["kimi_cli.tools.think:Think".to_string()],
                mcp_configs: Vec::new(),
            }),
            "Fixed coder".to_string(),
        )
        .expect("install fixed coder");

    let planner_definition = Arc::new(AgentDefinition {
        name: "Planner".to_string(),
        system_prompt: "You are a planner.".to_string(),
        tool_paths: vec!["kimi_cli.tools.multiagent:CreateSubagent".to_string()],
        mcp_configs: Vec::new(),
    });
    let tool = CreateSubagent::new(&fixture.runtime, planner_definition);

    let result = tool
        .call_typed(CreateSubagentParams {
            name: "coder".to_string(),
            system_prompt: "You are a dynamic coder.".to_string(),
        })
        .await;

    assert!(result.is_error);
    assert_eq!(result.message, "Subagent with name 'coder' already exists.");
}

#[tokio::test]
async fn test_dynamic_subagent_inherits_parent_external_tools_overlay() {
    let fixture = RuntimeFixture::new();
    let root_definition = agent_test_utils::test_agent_definition();
    let root_agent = root_definition
        .instantiate(fixture.runtime.clone())
        .await
        .expect("instantiate root agent");
    root_agent
        .toolset
        .lock()
        .await
        .register_external_tool(
            "wire_tool",
            "Wire provided tool",
            json!({
                "type": "object",
                "properties": {},
            }),
        )
        .expect("register external tool");

    let dynamic_definition = root_definition.derive_dynamic(
        "dynamic_agent".to_string(),
        "You are a dynamic agent.".to_string(),
    );
    let child_session = Session::create(
        fixture.runtime.storage.clone(),
        fixture.runtime.config.kaos.clone(),
        fixture.runtime.session.work_dir.clone(),
        Some("overlay-child".to_string()),
    )
    .await
    .expect("create child session");
    let child_runtime = fixture.runtime.rebind_session(child_session.clone());
    let overlay = root_agent.toolset.lock().await.snapshot_overlay();
    let child_agent = dynamic_definition
        .instantiate_with_overlay(child_runtime, &overlay)
        .await
        .expect("instantiate child agent with overlay");

    assert!(child_agent.toolset.lock().await.find("wire_tool").is_some());
}

#[tokio::test]
async fn test_task_resolves_global_fixed_subagent_from_registry() {
    let fixture = RuntimeFixture::new();
    fixture
        .runtime
        .subagent_registry
        .lock()
        .await
        .add_fixed_subagent(
            "coder".to_string(),
            Arc::new(AgentDefinition {
                name: "Coder".to_string(),
                system_prompt: "You are a coder.".to_string(),
                tool_paths: vec!["kimi_cli.tools.think:Think".to_string()],
                mcp_configs: Vec::new(),
            }),
            "Global coder".to_string(),
        )
        .expect("install fixed coder");

    let task = TaskTool::new(
        &fixture.runtime,
        Arc::new(tokio::sync::Mutex::new(
            kimi_agent::soul::toolset::KimiToolset::new(),
        )),
    );
    let result = task
        .call_typed(TaskParams {
            description: "Ask coder".to_string(),
            subagent_name: "coder".to_string(),
            prompt: "Solve this task".to_string(),
        })
        .await;

    assert!(result.is_error);
    assert_eq!(result.brief(), "Wire unavailable");
}
