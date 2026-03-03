mod agent_test_utils;
mod tool_test_utils;

use std::collections::HashMap;
use std::sync::Arc;

use tempfile::TempDir;

use kaos::KaosPath;
use kimi_agent::session::Session;
use kimi_agent::skill::flow::{Flow, FlowEdge, FlowLabel, FlowNode, FlowNodeKind};
use kimi_agent::skill::{Skill, SkillType};
use kimi_agent::soul::Soul;
use kimi_agent::soul::agent::Agent;
use kimi_agent::soul::context::Context;
use kimi_agent::soul::kimisoul::KimiSoul;
use kimi_agent::soul::toolset::KimiToolset;
use kimi_agent::soul::with_current_wire;
use kimi_agent::utils::SlashCommandInfo;
use kimi_agent::wire::UserInput;
use kimi_agent::wire::Wire;
use tool_test_utils::RuntimeFixture;

fn make_flow() -> Flow {
    let mut nodes = HashMap::new();
    nodes.insert(
        "BEGIN".to_string(),
        FlowNode::new("BEGIN", FlowLabel::from("Begin"), FlowNodeKind::Begin),
    );
    nodes.insert(
        "END".to_string(),
        FlowNode::new("END", FlowLabel::from("End"), FlowNodeKind::End),
    );

    let mut outgoing = HashMap::new();
    outgoing.insert(
        "BEGIN".to_string(),
        vec![FlowEdge::new("BEGIN", "END", None)],
    );
    outgoing.insert("END".to_string(), vec![]);

    Flow::new(nodes, outgoing, "BEGIN", "END")
}

#[test]
fn test_flow_skill_registers_skill_and_flow_commands() {
    let fixture = RuntimeFixture::new();
    let temp = TempDir::new().expect("temp dir");

    let flow = make_flow();
    let skill_dir = temp.path().join("flow-skill");
    std::fs::create_dir_all(&skill_dir).expect("create skill dir");
    let flow_skill = Skill {
        name: "flow-skill".to_string(),
        description: "Flow skill".to_string(),
        skill_type: SkillType::Flow,
        dir: KaosPath::unsafe_from_local_path(&skill_dir),
        flow: Some(flow),
        mcp_servers: Vec::new(),
    };

    let mut runtime = fixture.runtime.clone();
    runtime.skills = HashMap::from([("flow-skill".to_string(), flow_skill)]);
    let storage = runtime.storage.clone();
    let session_db_id = runtime.session.db_id();
    let session_id = runtime.session.id.clone();

    let agent = Agent {
        name: "Test Agent".to_string(),
        system_prompt: "Test system prompt.".to_string(),
        definition: agent_test_utils::test_agent_definition(),
        toolset: std::sync::Arc::new(tokio::sync::Mutex::new(KimiToolset::new())),
        runtime,
    };
    let soul = KimiSoul::new(agent, Context::new(storage, session_db_id, session_id));

    let command_names: std::collections::HashSet<String> = soul
        .available_slash_commands()
        .into_iter()
        .map(|cmd: SlashCommandInfo| cmd.name)
        .collect();

    assert!(command_names.contains("skill:flow-skill"));
    assert!(command_names.contains("flow:flow-skill"));
}

#[tokio::test]
async fn test_slash_only_turn_updates_session_title() {
    let fixture = RuntimeFixture::new();
    let storage = fixture.runtime.storage.clone();
    let session_db_id = fixture.runtime.session.db_id();
    let session_id = fixture.runtime.session.id.clone();
    let work_dir = fixture.runtime.session.work_dir.clone();
    let kaos = fixture.runtime.config.kaos.clone();

    let agent = Agent {
        name: "Test Agent".to_string(),
        system_prompt: "Test system prompt.".to_string(),
        definition: agent_test_utils::test_agent_definition(),
        toolset: std::sync::Arc::new(tokio::sync::Mutex::new(KimiToolset::new())),
        runtime: fixture.runtime.clone(),
    };
    let soul = KimiSoul::new(
        agent,
        Context::new(storage.clone(), session_db_id, session_id.clone()),
    );

    with_current_wire(Arc::new(Wire::new(None)), async {
        soul.run(UserInput::Text("/clear".to_string()))
            .await
            .expect("run slash command");
    })
    .await;

    let found = Session::find(storage, kaos, work_dir, &session_id)
        .await
        .expect("find session")
        .expect("session");
    assert!(found.title.starts_with("/clear"));
}
