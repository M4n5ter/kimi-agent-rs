use std::path::Path;
use std::sync::{Arc, Mutex};

use tempfile::TempDir;
use tokio::sync::Mutex as AsyncMutex;

use kaos::{
    AsyncReadWrite, CurrentKaosToken, ExecOptions, Kaos, KaosPath, KaosProcess, LineStream,
    LocalKaos, StrOrKaosPath, reset_current_kaos, set_current_kaos, with_current_kaos_scope,
};
use kimi_agent::config::get_default_config;
use kimi_agent::metadata::WorkDirMeta;
use kimi_agent::session::Session;
use kimi_agent::skill::{
    Skill, SkillMcpServer, SkillType, discover_skills, discover_skills_from_roots,
    find_user_skills_dir, resolve_skills_roots,
};
use kimi_agent::soul::agent::Runtime;
use kimi_agent::wire::WireFile;

static ENV_LOCK: AsyncMutex<()> = AsyncMutex::const_new(());

struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        // SAFETY: tests serialize env access via ENV_LOCK to avoid races.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev {
            // SAFETY: tests serialize env access via ENV_LOCK to avoid races.
            unsafe {
                std::env::set_var(self.key, prev);
            }
        } else {
            // SAFETY: tests serialize env access via ENV_LOCK to avoid races.
            unsafe {
                std::env::remove_var(self.key);
            }
        }
    }
}

struct ScopedKaos {
    inner: LocalKaos,
    backend_name: &'static str,
    root: KaosPath,
    home: KaosPath,
    cwd: Mutex<KaosPath>,
    fail_managed_dir_writes: bool,
}

impl ScopedKaos {
    fn new(
        backend_name: &'static str,
        root: KaosPath,
        home: KaosPath,
        cwd: KaosPath,
        fail_managed_dir_writes: bool,
    ) -> Self {
        Self {
            inner: LocalKaos::new(),
            backend_name,
            root,
            home,
            cwd: Mutex::new(cwd),
            fail_managed_dir_writes,
        }
    }

    fn resolve_path(&self, path: &KaosPath) -> KaosPath {
        let absolute = if path.is_absolute() {
            path.clone()
        } else {
            self.cwd.lock().unwrap().clone() / path
        };
        self.inner.normpath(&StrOrKaosPath::KaosPath(&absolute))
    }

    fn ensure_within_root(&self, path: &KaosPath) -> anyhow::Result<()> {
        let resolved = self.resolve_path(path);
        if resolved.relative_to(&self.root).is_ok() {
            return Ok(());
        }
        anyhow::bail!(
            "Path {} escapes test root {}",
            resolved.to_string_lossy(),
            self.root.to_string_lossy()
        );
    }

    fn ensure_managed_dir_writable(&self, path: &KaosPath) -> anyhow::Result<()> {
        if !self.fail_managed_dir_writes {
            return Ok(());
        }
        let resolved = self.resolve_path(path);
        let managed_dir = self.app_state_dir("kimi");
        if resolved.relative_to(&managed_dir).is_ok() {
            anyhow::bail!(
                "Managed app state dir is read-only: {}",
                managed_dir.to_string_lossy()
            );
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl Kaos for ScopedKaos {
    fn name(&self) -> &str {
        self.backend_name
    }

    fn platform(&self) -> kaos::KaosPlatform {
        self.inner.platform()
    }

    fn normpath(&self, path: &StrOrKaosPath<'_>) -> KaosPath {
        self.inner.normpath(path)
    }

    fn home(&self) -> KaosPath {
        self.home.clone()
    }

    fn cwd(&self) -> KaosPath {
        self.cwd.lock().unwrap().clone()
    }

    async fn chdir(&self, path: &KaosPath) -> anyhow::Result<()> {
        self.ensure_within_root(path)?;
        *self.cwd.lock().unwrap() = self.resolve_path(path);
        Ok(())
    }

    async fn stat(
        &self,
        path: &KaosPath,
        follow_symlinks: bool,
    ) -> anyhow::Result<kaos::StatResult> {
        self.ensure_within_root(path)?;
        self.inner.stat(path, follow_symlinks).await
    }

    async fn iterdir(&self, path: &KaosPath) -> anyhow::Result<Vec<KaosPath>> {
        self.ensure_within_root(path)?;
        self.inner.iterdir(path).await
    }

    async fn glob(
        &self,
        path: &KaosPath,
        pattern: &str,
        case_sensitive: bool,
    ) -> anyhow::Result<Vec<KaosPath>> {
        self.ensure_within_root(path)?;
        self.inner.glob(path, pattern, case_sensitive).await
    }

    async fn read_bytes(&self, path: &KaosPath, limit: Option<usize>) -> anyhow::Result<Vec<u8>> {
        self.ensure_within_root(path)?;
        self.inner.read_bytes(path, limit).await
    }

    async fn read_text(&self, path: &KaosPath) -> anyhow::Result<String> {
        self.ensure_within_root(path)?;
        self.inner.read_text(path).await
    }

    async fn read_lines(&self, path: &KaosPath) -> anyhow::Result<Vec<String>> {
        self.ensure_within_root(path)?;
        self.inner.read_lines(path).await
    }

    async fn read_lines_stream(&self, path: &KaosPath) -> anyhow::Result<LineStream> {
        self.ensure_within_root(path)?;
        self.inner.read_lines_stream(path).await
    }

    async fn write_bytes(&self, path: &KaosPath, data: &[u8]) -> anyhow::Result<usize> {
        self.ensure_within_root(path)?;
        self.ensure_managed_dir_writable(path)?;
        self.inner.write_bytes(path, data).await
    }

    async fn write_text(&self, path: &KaosPath, data: &str, append: bool) -> anyhow::Result<usize> {
        self.ensure_within_root(path)?;
        self.ensure_managed_dir_writable(path)?;
        self.inner.write_text(path, data, append).await
    }

    async fn chmod(&self, path: &KaosPath, mode: u32) -> anyhow::Result<()> {
        self.ensure_within_root(path)?;
        self.ensure_managed_dir_writable(path)?;
        self.inner.chmod(path, mode).await
    }

    async fn mkdir(&self, path: &KaosPath, parents: bool, exist_ok: bool) -> anyhow::Result<()> {
        self.ensure_within_root(path)?;
        self.ensure_managed_dir_writable(path)?;
        self.inner.mkdir(path, parents, exist_ok).await
    }

    async fn env_var(&self, key: &str) -> anyhow::Result<Option<String>> {
        self.inner.env_var(key).await
    }

    async fn exec(
        &self,
        args: &[String],
        options: ExecOptions,
    ) -> anyhow::Result<Box<dyn KaosProcess>> {
        self.inner.exec(args, options).await
    }

    async fn connect_tcp(&self, host: &str, port: u16) -> anyhow::Result<Box<dyn AsyncReadWrite>> {
        self.inner.connect_tcp(host, port).await
    }
}

struct FixedHomeKaosGuard {
    token: Option<CurrentKaosToken>,
}

impl FixedHomeKaosGuard {
    fn new(
        backend_name: &'static str,
        root: KaosPath,
        home: KaosPath,
        cwd: KaosPath,
        fail_managed_dir_writes: bool,
    ) -> Self {
        let kaos = Arc::new(ScopedKaos::new(
            backend_name,
            root,
            home,
            cwd,
            fail_managed_dir_writes,
        ));
        let token = set_current_kaos(kaos);
        Self { token: Some(token) }
    }
}

impl Drop for FixedHomeKaosGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            reset_current_kaos(token);
        }
    }
}

fn write_skill(skill_dir: &Path, content: &str) {
    std::fs::create_dir_all(skill_dir).expect("create skill dir");
    std::fs::write(skill_dir.join("SKILL.md"), content).expect("write skill");
}

fn assert_is_managed_builtin_root(root: &KaosPath, home: &KaosPath) {
    let managed_prefix = home.clone() / ".kimi" / "builtin-skills";
    assert!(
        root.relative_to(&managed_prefix).is_ok(),
        "expected {} to live under {}",
        root.to_string_lossy(),
        managed_prefix.to_string_lossy()
    );
}

fn build_test_session(work_dir: &KaosPath, share_dir: &Path) -> Session {
    let work_dir_meta = WorkDirMeta {
        path: work_dir.to_string_lossy(),
        kaos: "test".to_string(),
        last_session_id: None,
    };
    let context_file = share_dir.join("context.jsonl");
    std::fs::write(&context_file, "").expect("context file");

    Session {
        id: "skill-runtime".to_string(),
        work_dir: work_dir.clone(),
        work_dir_meta,
        context_file,
        wire_file: WireFile::new(share_dir.join("wire.jsonl")),
        title: "Skill Runtime".to_string(),
        updated_at: 0.0,
    }
}

#[tokio::test]
async fn test_discover_skills_parses_frontmatter_and_defaults() {
    let root = TempDir::new().expect("temp dir");
    let root_path = root.path().join("skills");
    std::fs::create_dir_all(&root_path).expect("create skills root");

    write_skill(
        &root_path.join("alpha"),
        "---\nname: alpha-skill\ndescription: Alpha description\n---\n",
    );
    write_skill(&root_path.join("beta"), "# No frontmatter");

    let root_path = KaosPath::unsafe_from_local_path(&root_path);
    let mut skills = discover_skills(&root_path).await;
    let base_dir = KaosPath::unsafe_from_local_path(Path::new("/path/to"));
    for skill in &mut skills {
        let relative_dir = skill.dir.relative_to(&root_path).expect("relative");
        skill.dir = base_dir.clone() / &relative_dir;
    }

    assert_eq!(
        skills,
        vec![
            Skill {
                name: "alpha-skill".to_string(),
                description: "Alpha description".to_string(),
                skill_type: SkillType::Standard,
                dir: KaosPath::unsafe_from_local_path(Path::new("/path/to/alpha")),
                flow: None,
                mcp_servers: Vec::new(),
            },
            Skill {
                name: "beta".to_string(),
                description: "No description provided.".to_string(),
                skill_type: SkillType::Standard,
                dir: KaosPath::unsafe_from_local_path(Path::new("/path/to/beta")),
                flow: None,
                mcp_servers: Vec::new(),
            },
        ]
    );
}

#[tokio::test]
async fn test_discover_skills_parses_flow_type() {
    let root = TempDir::new().expect("temp dir");
    let root_path = root.path().join("skills");
    std::fs::create_dir_all(&root_path).expect("create skills root");

    write_skill(
        &root_path.join("flowy"),
        "---\nname: flowy\ndescription: Flow skill\ntype: flow\n---\n```mermaid\nflowchart TD\nBEGIN([BEGIN]) --> A[Hello]\nA --> END([END])\n```\n",
    );

    let skills = discover_skills(&KaosPath::unsafe_from_local_path(&root_path)).await;

    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].skill_type, SkillType::Flow);
    assert!(skills[0].flow.is_some());
    assert_eq!(skills[0].flow.as_ref().unwrap().begin_id, "BEGIN");
}

#[tokio::test]
async fn test_discover_skills_parses_mcp_servers() {
    let root = TempDir::new().expect("temp dir");
    let root_path = root.path().join("skills");
    std::fs::create_dir_all(&root_path).expect("create skills root");

    write_skill(
        &root_path.join("mcp"),
        "---\nname: mcp\ndescription: MCP skill\nmcp:\n  - name: local\n    type: stdio\n    command: npx\n    args: [\"-y\", \"my-mcp\"]\n  - name: remote\n    type: http\n    url: https://example.com/mcp\n    transport: streamable-http\n---\n# body\n",
    );

    let skills = discover_skills(&KaosPath::unsafe_from_local_path(&root_path)).await;
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].mcp_servers.len(), 2);

    match &skills[0].mcp_servers[0] {
        SkillMcpServer::Stdio(server) => {
            assert_eq!(server.name, "local");
            assert_eq!(server.command, "npx");
            assert_eq!(server.args, vec!["-y".to_string(), "my-mcp".to_string()]);
        }
        _ => panic!("expected stdio mcp server"),
    }

    match &skills[0].mcp_servers[1] {
        SkillMcpServer::Http(server) => {
            assert_eq!(server.name, "remote");
            assert_eq!(server.url, "https://example.com/mcp");
            assert_eq!(server.transport.as_deref(), Some("streamable-http"));
        }
        _ => panic!("expected http mcp server"),
    }
}

#[tokio::test]
async fn test_discover_skills_rejects_duplicate_mcp_server_names() {
    let root = TempDir::new().expect("temp dir");
    let root_path = root.path().join("skills");
    std::fs::create_dir_all(&root_path).expect("create skills root");

    write_skill(
        &root_path.join("dup-mcp"),
        "---\nname: dup\ndescription: duplicated mcp names\nmcp:\n  - name: same\n    type: stdio\n    command: cmd-a\n  - name: SAME\n    type: stdio\n    command: cmd-b\n---\n# body\n",
    );

    let skills = discover_skills(&KaosPath::unsafe_from_local_path(&root_path)).await;
    assert!(skills.is_empty());
}

#[tokio::test]
async fn test_discover_skills_flow_parse_failure_falls_back() {
    let root = TempDir::new().expect("temp dir");
    let root_path = root.path().join("skills");
    std::fs::create_dir_all(&root_path).expect("create skills root");

    write_skill(
        &root_path.join("broken-flow"),
        "---\nname: broken-flow\ndescription: Broken flow skill\ntype: flow\n---\n```mermaid\nflowchart TD\nA --> B\n```\n",
    );

    let skills = discover_skills(&KaosPath::unsafe_from_local_path(&root_path)).await;

    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].skill_type, SkillType::Standard);
    assert!(skills[0].flow.is_none());
}

#[tokio::test]
async fn test_discover_skills_from_roots_prefers_later_dirs() {
    let root = TempDir::new().expect("temp dir");
    let root_path = root.path().join("root");
    let system_dir = root_path.join("system");
    let user_dir = root_path.join("user");
    std::fs::create_dir_all(&system_dir).expect("create system dir");
    std::fs::create_dir_all(&user_dir).expect("create user dir");

    write_skill(
        &system_dir.join("shared"),
        "---\nname: shared\ndescription: System version\n---\n",
    );
    write_skill(
        &user_dir.join("shared"),
        "---\nname: shared\ndescription: User version\n---\n",
    );

    let root_path = KaosPath::unsafe_from_local_path(&root_path);
    let mut skills = discover_skills_from_roots(&[
        KaosPath::unsafe_from_local_path(&system_dir),
        KaosPath::unsafe_from_local_path(&user_dir),
    ])
    .await;
    let base_dir = KaosPath::unsafe_from_local_path(Path::new("/path/to"));
    for skill in &mut skills {
        let relative_dir = skill.dir.relative_to(&root_path).expect("relative");
        skill.dir = base_dir.clone() / &relative_dir;
    }

    assert_eq!(
        skills,
        vec![Skill {
            name: "shared".to_string(),
            description: "User version".to_string(),
            skill_type: SkillType::Standard,
            dir: KaosPath::unsafe_from_local_path(Path::new("/path/to/user/shared")),
            flow: None,
            mcp_servers: Vec::new(),
        }]
    );
}

#[tokio::test]
async fn test_resolve_skills_roots_uses_layers() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("project");
        std::fs::create_dir_all(&home_dir).expect("create home dir");
        std::fs::create_dir_all(&work_dir).expect("create work dir");
        let root_path = KaosPath::unsafe_from_local_path(&root_dir);
        let home_path = KaosPath::unsafe_from_local_path(&home_dir);
        let work_path = KaosPath::unsafe_from_local_path(&work_dir);
        let _kaos_guard = FixedHomeKaosGuard::new(
            "local",
            root_path,
            home_path.clone(),
            work_path.clone(),
            false,
        );
        let user_dir = home_dir.join(".config/agents/skills");
        std::fs::create_dir_all(&user_dir).expect("create user skills dir");

        let project_dir = work_dir.join(".agents/skills");
        std::fs::create_dir_all(&project_dir).expect("create project skills dir");

        let roots = resolve_skills_roots(&work_path, None).await;

        assert_eq!(roots.len(), 3);
        assert_is_managed_builtin_root(&roots[0], &home_path);
        assert_eq!(roots[1], KaosPath::unsafe_from_local_path(&user_dir));
        assert_eq!(roots[2], KaosPath::unsafe_from_local_path(&project_dir));
    })
    .await;
}

#[tokio::test]
async fn test_resolve_skills_roots_uses_kimi_share_dir_for_local_backend() {
    let _lock = ENV_LOCK.lock().await;
    let tmp = TempDir::new().expect("temp dir");
    let home_dir = tmp.path().join("home");
    let share_dir = tmp.path().join("share");
    let work_dir = tmp.path().join("work");
    std::fs::create_dir_all(&home_dir).expect("create home dir");
    std::fs::create_dir_all(&share_dir).expect("create share dir");
    std::fs::create_dir_all(&work_dir).expect("create work dir");
    let _home = EnvGuard::set("HOME", home_dir.to_str().expect("home path"));
    let _userprofile = EnvGuard::set("USERPROFILE", home_dir.to_str().expect("home path"));
    let _share = EnvGuard::set("KIMI_SHARE_DIR", share_dir.to_str().expect("share path"));

    with_current_kaos_scope(async {
        let token = set_current_kaos(Arc::new(LocalKaos::new()));
        let work_path = KaosPath::unsafe_from_local_path(&work_dir);
        let roots = resolve_skills_roots(&work_path, None).await;
        reset_current_kaos(token);

        assert!(!roots.is_empty());
        let managed_prefix = KaosPath::unsafe_from_local_path(&share_dir) / "builtin-skills";
        assert!(
            roots[0].relative_to(&managed_prefix).is_ok(),
            "expected {} to live under {}",
            roots[0].to_string_lossy(),
            managed_prefix.to_string_lossy()
        );
    })
    .await;
}

#[tokio::test]
async fn test_resolve_skills_roots_respects_override() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("project");
        let override_dir = work_dir.join("override");
        std::fs::create_dir_all(&home_dir).expect("create home dir");
        std::fs::create_dir_all(&override_dir).expect("create override dir");

        let root_path = KaosPath::unsafe_from_local_path(&root_dir);
        let work_path = KaosPath::unsafe_from_local_path(&work_dir);
        let _kaos_guard = FixedHomeKaosGuard::new(
            "local",
            root_path,
            KaosPath::unsafe_from_local_path(&home_dir),
            work_path.clone(),
            false,
        );

        let roots = resolve_skills_roots(
            &work_path,
            Some(KaosPath::unsafe_from_local_path(&override_dir)),
        )
        .await;

        assert_eq!(roots, vec![KaosPath::unsafe_from_local_path(&override_dir)]);
    })
    .await;
}

#[tokio::test]
async fn test_find_user_skills_dir_uses_agents_candidate() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("work");
        std::fs::create_dir_all(&home_dir).expect("create home dir");
        std::fs::create_dir_all(&work_dir).expect("create work dir");
        let _kaos_guard = FixedHomeKaosGuard::new(
            "local",
            KaosPath::unsafe_from_local_path(&root_dir),
            KaosPath::unsafe_from_local_path(&home_dir),
            KaosPath::unsafe_from_local_path(&work_dir),
            false,
        );

        let agents_dir = home_dir.join(".agents/skills");
        std::fs::create_dir_all(&agents_dir).expect("create agents skills dir");

        let found = find_user_skills_dir().await.expect("user skills dir");
        assert_eq!(found, KaosPath::unsafe_from_local_path(&agents_dir));
    })
    .await;
}

#[tokio::test]
async fn test_find_user_skills_dir_uses_codex_candidate() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("work");
        std::fs::create_dir_all(&home_dir).expect("create home dir");
        std::fs::create_dir_all(&work_dir).expect("create work dir");
        let _kaos_guard = FixedHomeKaosGuard::new(
            "local",
            KaosPath::unsafe_from_local_path(&root_dir),
            KaosPath::unsafe_from_local_path(&home_dir),
            KaosPath::unsafe_from_local_path(&work_dir),
            false,
        );

        let codex_dir = home_dir.join(".codex/skills");
        std::fs::create_dir_all(&codex_dir).expect("create codex skills dir");

        let found = find_user_skills_dir().await.expect("user skills dir");
        assert_eq!(found, KaosPath::unsafe_from_local_path(&codex_dir));
    })
    .await;
}

#[tokio::test]
async fn test_resolve_skills_roots_materializes_builtin_skills_for_remote_backend() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("remote-root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("work");
        std::fs::create_dir_all(&home_dir).expect("create home dir");
        std::fs::create_dir_all(&work_dir).expect("create work dir");

        let root_path = KaosPath::unsafe_from_local_path(&root_dir);
        let home_path = KaosPath::unsafe_from_local_path(&home_dir);
        let work_path = KaosPath::unsafe_from_local_path(&work_dir);
        let _kaos_guard = FixedHomeKaosGuard::new(
            "ssh",
            root_path,
            home_path.clone(),
            work_path.clone(),
            false,
        );

        let roots = resolve_skills_roots(&work_path, None).await;
        assert!(!roots.is_empty());
        assert_is_managed_builtin_root(&roots[0], &home_path);

        let skills = discover_skills_from_roots(&roots).await;
        let names: Vec<_> = skills.iter().map(|skill| skill.name.as_str()).collect();
        assert!(names.contains(&"kimi-cli-help"));
        assert!(names.contains(&"skill-creator"));
    })
    .await;
}

#[tokio::test]
async fn test_runtime_create_lists_managed_builtin_skill_paths() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("remote-root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("work");
        let share_dir = tmp.path().join("share");
        std::fs::create_dir_all(&home_dir).expect("create home dir");
        std::fs::create_dir_all(&work_dir).expect("create work dir");
        std::fs::create_dir_all(&share_dir).expect("create share dir");

        let root_path = KaosPath::unsafe_from_local_path(&root_dir);
        let home_path = KaosPath::unsafe_from_local_path(&home_dir);
        let work_path = KaosPath::unsafe_from_local_path(&work_dir);
        let _kaos_guard =
            FixedHomeKaosGuard::new("ssh", root_path, home_path, work_path.clone(), false);

        let session = build_test_session(&work_path, &share_dir);
        let runtime = Runtime::create(get_default_config(), None, session, true, None).await;

        assert!(
            runtime
                .builtin_args
                .KIMI_SKILLS
                .contains(".kimi/builtin-skills/")
        );
        assert!(
            runtime
                .builtin_args
                .KIMI_SKILLS
                .contains("skill-creator/SKILL.md")
        );
    })
    .await;
}

#[tokio::test]
async fn test_resolve_skills_roots_skips_builtin_when_managed_dir_is_unwritable() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("remote-root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("work");
        let user_dir = home_dir.join(".config/agents/skills");
        let project_dir = work_dir.join(".agents/skills");
        std::fs::create_dir_all(&user_dir).expect("create user dir");
        std::fs::create_dir_all(&project_dir).expect("create project dir");

        let root_path = KaosPath::unsafe_from_local_path(&root_dir);
        let home_path = KaosPath::unsafe_from_local_path(&home_dir);
        let work_path = KaosPath::unsafe_from_local_path(&work_dir);
        let _kaos_guard =
            FixedHomeKaosGuard::new("ssh", root_path, home_path, work_path.clone(), true);

        let roots = resolve_skills_roots(&work_path, None).await;
        assert_eq!(
            roots,
            vec![
                KaosPath::unsafe_from_local_path(&user_dir),
                KaosPath::unsafe_from_local_path(&project_dir),
            ]
        );
    })
    .await;
}

#[tokio::test]
async fn test_resolve_skills_roots_reuses_existing_read_only_builtin_bundle() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("remote-root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("work");
        std::fs::create_dir_all(&home_dir).expect("create home dir");
        std::fs::create_dir_all(&work_dir).expect("create work dir");

        let root_path = KaosPath::unsafe_from_local_path(&root_dir);
        let home_path = KaosPath::unsafe_from_local_path(&home_dir);
        let work_path = KaosPath::unsafe_from_local_path(&work_dir);

        let builtin_root = {
            let _kaos_guard = FixedHomeKaosGuard::new(
                "ssh",
                root_path.clone(),
                home_path.clone(),
                work_path.clone(),
                false,
            );
            let roots = resolve_skills_roots(&work_path, None).await;
            assert!(!roots.is_empty());
            roots[0].clone()
        };

        let ready_marker = builtin_root
            .clone()
            .unsafe_to_local_path()
            .join(".bundle-ready");
        std::fs::remove_file(&ready_marker).expect("remove ready marker");

        let _kaos_guard = FixedHomeKaosGuard::new("ssh", root_path, home_path, work_path, true);
        let roots = resolve_skills_roots(
            &KaosPath::unsafe_from_local_path(&root_dir.join("work")),
            None,
        )
        .await;
        assert!(!roots.is_empty());
        assert_eq!(roots[0], builtin_root);

        let skills = discover_skills_from_roots(&roots).await;
        let names: Vec<_> = skills.iter().map(|skill| skill.name.as_str()).collect();
        assert!(names.contains(&"kimi-cli-help"));
        assert!(names.contains(&"skill-creator"));
    })
    .await;
}

#[tokio::test]
async fn test_override_bypasses_builtin_sync_when_managed_dir_is_unwritable() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("remote-root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("work");
        let override_dir = root_dir.join("override");
        std::fs::create_dir_all(&home_dir).expect("create home dir");
        std::fs::create_dir_all(&work_dir).expect("create work dir");
        std::fs::create_dir_all(&override_dir).expect("create override dir");

        let root_path = KaosPath::unsafe_from_local_path(&root_dir);
        let home_path = KaosPath::unsafe_from_local_path(&home_dir);
        let work_path = KaosPath::unsafe_from_local_path(&work_dir);
        let _kaos_guard = FixedHomeKaosGuard::new("ssh", root_path, home_path, work_path, true);

        let roots = resolve_skills_roots(
            &KaosPath::unsafe_from_local_path(&root_dir),
            Some(KaosPath::unsafe_from_local_path(&override_dir)),
        )
        .await;
        assert_eq!(roots, vec![KaosPath::unsafe_from_local_path(&override_dir)]);
    })
    .await;
}

#[tokio::test]
async fn test_runtime_create_continues_without_builtin_skills_when_managed_dir_is_unwritable() {
    with_current_kaos_scope(async {
        let tmp = TempDir::new().expect("temp dir");
        let root_dir = tmp.path().join("remote-root");
        let home_dir = root_dir.join("home");
        let work_dir = root_dir.join("work");
        let share_dir = tmp.path().join("share");
        std::fs::create_dir_all(&home_dir).expect("create home dir");
        std::fs::create_dir_all(&work_dir).expect("create work dir");
        std::fs::create_dir_all(&share_dir).expect("create share dir");

        let root_path = KaosPath::unsafe_from_local_path(&root_dir);
        let home_path = KaosPath::unsafe_from_local_path(&home_dir);
        let work_path = KaosPath::unsafe_from_local_path(&work_dir);
        let _kaos_guard =
            FixedHomeKaosGuard::new("ssh", root_path, home_path, work_path.clone(), true);

        let session = build_test_session(&work_path, &share_dir);
        let runtime = Runtime::create(get_default_config(), None, session, true, None).await;
        assert_eq!(runtime.builtin_args.KIMI_SKILLS, "No skills found.");
    })
    .await;
}
