mod agent_test_utils;
mod tool_test_utils;

use tempfile::TempDir;

use kimi_agent::exception::SystemPromptTemplateError;
use kimi_agent::soul::agent::load_agent;
use kimi_agent::soul::toolset::KimiToolset;
use kosong::tooling::Toolset;

use tool_test_utils::RuntimeFixture;

fn write_file(path: &std::path::Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent");
    }
    std::fs::write(path, content).expect("write file");
}

#[tokio::test]
async fn test_load_system_prompt_substitution() {
    let fixture = RuntimeFixture::new();

    let dir = TempDir::new().expect("temp dir");
    let system_md = dir.path().join("system.md");
    write_file(
        &system_md,
        "Test system prompt with ${KIMI_NOW} and ${CUSTOM_ARG}",
    );

    let agent_yaml = dir.path().join("agent.yaml");
    write_file(
        &agent_yaml,
        r#"version: 1
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
  system_prompt_args:
    CUSTOM_ARG: "test_value"
  tools: ["kimi_cli.tools.think:Think"]
"#,
    );

    let agent = load_agent(&agent_yaml, fixture.runtime.clone(), &[])
        .await
        .expect("load agent");

    assert!(agent.system_prompt.contains("Test system prompt with"));
    assert!(
        agent
            .system_prompt
            .contains(&fixture.runtime.builtin_args.KIMI_NOW)
    );
    assert!(agent.system_prompt.contains("test_value"));
}

#[tokio::test]
async fn test_load_system_prompt_allows_literal_dollar() {
    let fixture = RuntimeFixture::new();

    let dir = TempDir::new().expect("temp dir");
    let system_md = dir.path().join("system.md");
    write_file(&system_md, "Price is $100, path $PATH, time ${KIMI_NOW}.");

    let agent_yaml = dir.path().join("agent.yaml");
    write_file(
        &agent_yaml,
        r#"version: 1
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.think:Think"]
"#,
    );

    let agent = load_agent(&agent_yaml, fixture.runtime.clone(), &[])
        .await
        .expect("load agent");

    assert!(agent.system_prompt.contains("$100"));
    assert!(agent.system_prompt.contains("$PATH"));
    assert!(
        agent
            .system_prompt
            .contains(&fixture.runtime.builtin_args.KIMI_NOW)
    );
}

#[tokio::test]
async fn test_load_system_prompt_missing_arg_raises() {
    let fixture = RuntimeFixture::new();

    let dir = TempDir::new().expect("temp dir");
    let system_md = dir.path().join("system.md");
    write_file(&system_md, "Missing ${UNKNOWN_ARG}.");

    let agent_yaml = dir.path().join("agent.yaml");
    write_file(
        &agent_yaml,
        r#"version: 1
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.think:Think"]
"#,
    );

    let err = match load_agent(&agent_yaml, fixture.runtime.clone(), &[]).await {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };
    assert!(err.downcast_ref::<SystemPromptTemplateError>().is_some());
}

#[test]
fn test_load_tools_valid() {
    let fixture = RuntimeFixture::new();
    let tool_paths = vec![
        "kimi_cli.tools.think:Think".to_string(),
        "kimi_cli.tools.shell:Shell".to_string(),
    ];
    let mut toolset = KimiToolset::new();
    let deps_toolset = std::sync::Arc::new(tokio::sync::Mutex::new(KimiToolset::new()));
    toolset
        .load_tools(
            &tool_paths,
            &fixture.runtime,
            deps_toolset,
            agent_test_utils::test_agent_definition(),
        )
        .expect("load tools");
    assert_eq!(toolset.tools().len(), 2);
}

#[test]
fn test_load_tools_invalid() {
    let fixture = RuntimeFixture::new();
    let tool_paths = vec![
        "kimi_cli.tools.nonexistent:Tool".to_string(),
        "kimi_cli.tools.think:Think".to_string(),
    ];
    let mut toolset = KimiToolset::new();
    let deps_toolset = std::sync::Arc::new(tokio::sync::Mutex::new(KimiToolset::new()));
    let result = toolset.load_tools(
        &tool_paths,
        &fixture.runtime,
        deps_toolset,
        agent_test_utils::test_agent_definition(),
    );
    let err = result.expect_err("expected error");
    assert!(err.to_string().contains("kimi_cli.tools.nonexistent:Tool"));
}

#[tokio::test]
async fn test_load_agent_invalid_tools() {
    let fixture = RuntimeFixture::new();

    let dir = TempDir::new().expect("temp dir");
    let system_md = dir.path().join("system.md");
    write_file(&system_md, "You are a test agent");

    let agent_yaml = dir.path().join("agent.yaml");
    write_file(
        &agent_yaml,
        r#"version: 1
agent:
  name: "Test Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.nonexistent:Tool"]
"#,
    );

    match load_agent(&agent_yaml, fixture.runtime.clone(), &[]).await {
        Ok(_) => panic!("expected error"),
        Err(err) => assert!(err.to_string().contains("Invalid tools")),
    }
}

#[tokio::test]
async fn test_load_agent_invalid_fixed_subagent_tools() {
    let fixture = RuntimeFixture::new();

    let dir = TempDir::new().expect("temp dir");
    let system_md = dir.path().join("system.md");
    write_file(&system_md, "You are a test agent");

    let broken_system_md = dir.path().join("broken-system.md");
    write_file(&broken_system_md, "You are a broken subagent");

    let broken_agent_yaml = dir.path().join("broken-agent.yaml");
    write_file(
        &broken_agent_yaml,
        r#"version: 1
agent:
  name: "Broken Subagent"
  system_prompt_path: ./broken-system.md
  tools: ["kimi_cli.tools.nonexistent:Tool"]
"#,
    );

    let agent_yaml = dir.path().join("agent.yaml");
    write_file(
        &agent_yaml,
        r#"version: 1
agent:
  name: "Root Agent"
  system_prompt_path: ./system.md
  tools: ["kimi_cli.tools.think:Think"]
  subagents:
    broken:
      description: "Broken fixed subagent"
      path: ./broken-agent.yaml
"#,
    );

    match load_agent(&agent_yaml, fixture.runtime.clone(), &[]).await {
        Ok(_) => panic!("expected error"),
        Err(err) => assert!(err.to_string().contains("Invalid tools")),
    }
}

#[tokio::test]
async fn test_load_agent_registers_fixed_subagents_globally() {
    let fixture = RuntimeFixture::new();

    let dir = TempDir::new().expect("temp dir");
    write_file(&dir.path().join("root.md"), "Root");
    write_file(&dir.path().join("planner.md"), "Planner");
    write_file(&dir.path().join("coder.md"), "Coder");

    write_file(
        &dir.path().join("planner.yaml"),
        r#"version: 1
agent:
  name: "Planner"
  system_prompt_path: ./planner.md
  tools: ["kimi_cli.tools.think:Think"]
"#,
    );
    write_file(
        &dir.path().join("coder.yaml"),
        r#"version: 1
agent:
  name: "Coder"
  system_prompt_path: ./coder.md
  tools: ["kimi_cli.tools.think:Think"]
"#,
    );
    write_file(
        &dir.path().join("agent.yaml"),
        r#"version: 1
agent:
  name: "Root"
  system_prompt_path: ./root.md
  tools: ["kimi_cli.tools.multiagent:Task", "kimi_cli.tools.think:Think"]
  subagents:
    planner:
      description: "Planner fixed subagent"
      path: ./planner.yaml
    coder:
      description: "Coder fixed subagent"
      path: ./coder.yaml
"#,
    );

    let _agent = load_agent(&dir.path().join("agent.yaml"), fixture.runtime.clone(), &[])
        .await
        .expect("load agent");
    let registry = fixture.runtime.subagent_registry.lock().await;
    assert!(registry.contains("planner"));
    assert!(registry.contains("coder"));
}

#[tokio::test]
async fn test_load_agent_rejects_duplicate_fixed_subagent_names_across_tree() {
    let fixture = RuntimeFixture::new();

    let dir = TempDir::new().expect("temp dir");
    write_file(&dir.path().join("root.md"), "Root");
    write_file(&dir.path().join("planner.md"), "Planner");
    write_file(&dir.path().join("coder.md"), "Coder");
    write_file(&dir.path().join("nested-coder.md"), "Nested coder");

    write_file(
        &dir.path().join("nested-coder.yaml"),
        r#"version: 1
agent:
  name: "Nested Coder"
  system_prompt_path: ./nested-coder.md
  tools: ["kimi_cli.tools.think:Think"]
"#,
    );
    write_file(
        &dir.path().join("planner.yaml"),
        r#"version: 1
agent:
  name: "Planner"
  system_prompt_path: ./planner.md
  tools: ["kimi_cli.tools.think:Think"]
  subagents:
    coder:
      description: "Nested coder"
      path: ./nested-coder.yaml
"#,
    );
    write_file(
        &dir.path().join("coder.yaml"),
        r#"version: 1
agent:
  name: "Coder"
  system_prompt_path: ./coder.md
  tools: ["kimi_cli.tools.think:Think"]
"#,
    );
    write_file(
        &dir.path().join("agent.yaml"),
        r#"version: 1
agent:
  name: "Root"
  system_prompt_path: ./root.md
  tools: ["kimi_cli.tools.multiagent:Task", "kimi_cli.tools.think:Think"]
  subagents:
    planner:
      description: "Planner fixed subagent"
      path: ./planner.yaml
    coder:
      description: "Coder fixed subagent"
      path: ./coder.yaml
"#,
    );

    match load_agent(&dir.path().join("agent.yaml"), fixture.runtime.clone(), &[]).await {
        Ok(_) => panic!("expected duplicate fixed subagent error"),
        Err(err) => assert!(
            err.to_string()
                .contains("Duplicate fixed subagent name: coder")
        ),
    }
}
