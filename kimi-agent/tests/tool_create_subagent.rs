mod agent_test_utils;
mod tool_test_utils;

use kaos::KaosPath;
use kimi_agent::config::{LLMModel, LLMProvider, ProviderType, StorageConfig, get_default_config};
use kimi_agent::session::Session;
use kimi_agent::soul::agent::AgentDefinition;
use kimi_agent::tools::multiagent::{CreateSubagent, CreateSubagentParams, TaskParams, TaskTool};
use kosong::chat_provider::kimi::Kimi;
use kosong::tooling::CallableTool2;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

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
    let child_runtime = fixture
        .runtime
        .create_child_runtime(child_session.clone())
        .await
        .expect("create child runtime");

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
    let child_runtime = fixture
        .runtime
        .create_child_runtime(child_session.clone())
        .await
        .expect("create child runtime");
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

#[tokio::test]
async fn test_child_runtime_recreates_kimi_prompt_cache_key() {
    let storage_dir = TempDir::new().expect("storage dir");
    let work_dir = TempDir::new().expect("work dir");

    let mut config = get_default_config();
    config.storage = StorageConfig {
        database_path: storage_dir.path().join("state.db").display().to_string(),
        busy_timeout_ms: 1_000,
    };
    config.default_model = "kimi-test".to_string();
    config.models.insert(
        "kimi-test".to_string(),
        LLMModel {
            provider: "kimi-provider".to_string(),
            model: "kimi-base".to_string(),
            max_context_size: 4096,
            capabilities: None,
        },
    );
    config.providers.insert(
        "kimi-provider".to_string(),
        LLMProvider {
            provider_type: ProviderType::Kimi,
            base_url: "https://api.test/v1".to_string(),
            api_key: "test-key".to_string(),
            env: None,
            custom_headers: None,
        },
    );

    let storage = kimi_agent::storage::Storage::open(&config.storage)
        .await
        .expect("open storage");
    let root_session = Session::create(
        storage.clone(),
        config.kaos.clone(),
        KaosPath::from(PathBuf::from(work_dir.path())),
        Some("root".to_string()),
    )
    .await
    .expect("create root session");
    let runtime = kimi_agent::soul::agent::Runtime::create(
        config,
        storage.clone(),
        kimi_agent::llm::create_llm(
            &LLMProvider {
                provider_type: ProviderType::Kimi,
                base_url: "https://api.test/v1".to_string(),
                api_key: "test-key".to_string(),
                env: None,
                custom_headers: None,
            },
            &LLMModel {
                provider: "kimi-provider".to_string(),
                model: "kimi-base".to_string(),
                max_context_size: 4096,
                capabilities: None,
            },
            Some(false),
            Some(&root_session.id),
        )
        .await
        .expect("create root llm")
        .map(Arc::new),
        root_session.clone(),
        true,
        None,
    )
    .await;

    let child_session = Session::create(
        storage,
        runtime.config.kaos.clone(),
        root_session.work_dir.clone(),
        Some("child".to_string()),
    )
    .await
    .expect("create child session");
    let child_runtime = runtime
        .create_child_runtime(child_session.clone())
        .await
        .expect("create child runtime");

    let root_kimi = runtime
        .llm
        .as_ref()
        .expect("root llm")
        .chat_provider
        .as_any()
        .downcast_ref::<Kimi>()
        .expect("kimi provider");
    let child_kimi = child_runtime
        .llm
        .as_ref()
        .expect("child llm")
        .chat_provider
        .as_any()
        .downcast_ref::<Kimi>()
        .expect("kimi provider");

    assert_eq!(
        root_kimi.model_parameters().get("prompt_cache_key"),
        Some(&json!(root_session.id))
    );
    assert_eq!(
        child_kimi.model_parameters().get("prompt_cache_key"),
        Some(&json!(child_session.id))
    );
}

#[tokio::test]
async fn test_isolated_runtime_keeps_fixed_registry_only() {
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
    fixture
        .runtime
        .subagent_registry
        .lock()
        .await
        .add_dynamic_subagent(
            "dynamic".to_string(),
            Arc::new(AgentDefinition {
                name: "Dynamic".to_string(),
                system_prompt: "You are dynamic.".to_string(),
                tool_paths: vec!["kimi_cli.tools.think:Think".to_string()],
                mcp_configs: Vec::new(),
            }),
        );

    let tmp_session = Session::create(
        fixture.runtime.storage.clone(),
        fixture.runtime.config.kaos.clone(),
        fixture.runtime.session.work_dir.clone(),
        Some("tmp-init".to_string()),
    )
    .await
    .expect("create tmp session");
    let isolated = fixture
        .runtime
        .create_isolated_runtime(tmp_session)
        .await
        .expect("create isolated runtime");

    assert!(!Arc::ptr_eq(
        &fixture.runtime.subagent_registry,
        &isolated.subagent_registry
    ));
    let registry = isolated.subagent_registry.lock().await;
    assert!(registry.contains("coder"));
    assert!(!registry.contains("dynamic"));
}
