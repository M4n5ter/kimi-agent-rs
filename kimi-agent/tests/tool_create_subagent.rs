mod agent_test_utils;
mod tool_test_utils;

use kimi_agent::session::Session;
use kimi_agent::tools::multiagent::{CreateSubagent, CreateSubagentParams};
use kosong::tooling::CallableTool2;

use tool_test_utils::RuntimeFixture;

#[tokio::test]
async fn test_create_subagent() {
    let fixture = RuntimeFixture::new();
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
        .labor_market
        .lock()
        .await
        .all_dynamic_subagents();
    assert!(subagents.contains_key("test_agent"));
}

#[tokio::test]
async fn test_create_existing_subagent() {
    let fixture = RuntimeFixture::new();
    let tool = CreateSubagent::new(&fixture.runtime, agent_test_utils::test_agent_definition());

    let _ = tool
        .call_typed(CreateSubagentParams {
            name: "existing_agent".to_string(),
            system_prompt: "You are an existing agent.".to_string(),
        })
        .await;

    let subagents = fixture
        .runtime
        .labor_market
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
        .labor_market
        .lock()
        .await
        .all_dynamic_subagents();
    assert!(subagents.contains_key("existing_agent"));
}

#[tokio::test]
async fn test_created_subagent_instantiates_with_child_session_runtime() {
    let fixture = RuntimeFixture::new();
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
        .labor_market
        .lock()
        .await
        .all_dynamic_subagents()
        .remove("nested_agent")
        .expect("created subagent definition");
    assert!(definition.fixed_subagents.contains_key("mocker"));

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
