use anyhow::{Context, Result};
use kaos::{KaosPath, get_current_kaos};
use tracing::debug;

#[derive(Clone, Copy)]
struct EmbeddedFile {
    relative_path: &'static str,
    contents: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/builtin_skills.rs"));

const READY_MARKER_FILE: &str = ".bundle-ready";

pub async fn prepare_builtin_skills_root() -> Result<KaosPath> {
    let bundle_id = builtin_bundle_id();
    let root = builtin_bundle_root(&bundle_id);
    if is_materialized_bundle_usable(&root, &bundle_id, BUILTIN_SKILL_FILES).await? {
        debug!(
            root = %root.to_string_lossy(),
            files = BUILTIN_SKILL_FILES.len(),
            "Reusing builtin skills bundle"
        );
        return Ok(root);
    }

    materialize_embedded_files(&root, BUILTIN_SKILL_FILES).await?;
    write_ready_marker(&root, &bundle_id).await?;
    debug!(
        root = %root.to_string_lossy(),
        files = BUILTIN_SKILL_FILES.len(),
        "Materialized builtin skills bundle"
    );
    Ok(root)
}

fn builtin_bundle_id() -> String {
    let mut data = Vec::new();
    for file in BUILTIN_SKILL_FILES {
        data.extend_from_slice(file.relative_path.as_bytes());
        data.push(0);
        data.extend_from_slice(file.contents);
        data.push(0);
    }
    format!("{:x}", md5::compute(data))
}

fn builtin_bundle_root(bundle_id: &str) -> KaosPath {
    get_current_kaos().app_state_dir("kimi") / "builtin-skills" / bundle_id
}

fn ready_marker_path(root: &KaosPath) -> KaosPath {
    root.clone() / READY_MARKER_FILE
}

async fn is_materialized_bundle_usable(
    root: &KaosPath,
    bundle_id: &str,
    files: &[EmbeddedFile],
) -> Result<bool> {
    if !root.is_dir(true).await {
        return Ok(false);
    }

    let ready_marker = ready_marker_path(root);
    if let Ok(marker) = ready_marker.read_text().await
        && marker.trim() == bundle_id
    {
        return Ok(true);
    }

    let platform = get_current_kaos().platform();
    for file in files {
        let path = path_from_relative(root, file.relative_path);
        let contents = match path.read_bytes(None).await {
            Ok(contents) => contents,
            Err(_) => return Ok(false),
        };
        if contents != file.contents {
            return Ok(false);
        }
        if platform.os != "windows" && is_script_asset(file.relative_path) {
            let stat = match path.stat(true).await {
                Ok(stat) => stat,
                Err(_) => return Ok(false),
            };
            if stat.st_mode & 0o111 == 0 {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

async fn materialize_embedded_files(root: &KaosPath, files: &[EmbeddedFile]) -> Result<()> {
    root.mkdir(true, true).await.with_context(|| {
        format!(
            "Failed to create builtin skills root {}",
            root.to_string_lossy()
        )
    })?;

    let platform = get_current_kaos().platform();
    for file in files {
        let path = path_from_relative(root, file.relative_path);
        let parent = path.parent();
        parent.mkdir(true, true).await.with_context(|| {
            format!(
                "Failed to create parent directory {}",
                parent.to_string_lossy()
            )
        })?;
        path.write_bytes(file.contents).await.with_context(|| {
            format!(
                "Failed to write builtin skill asset {}",
                path.to_string_lossy()
            )
        })?;
        if platform.os != "windows" && is_script_asset(file.relative_path) {
            path.chmod(0o755).await.with_context(|| {
                format!("Failed to chmod builtin script {}", path.to_string_lossy())
            })?;
        }
    }

    Ok(())
}

async fn write_ready_marker(root: &KaosPath, bundle_id: &str) -> Result<()> {
    let ready_marker = ready_marker_path(root);
    ready_marker.write_text(bundle_id).await.with_context(|| {
        format!(
            "Failed to write builtin bundle marker {}",
            ready_marker.to_string_lossy()
        )
    })?;
    Ok(())
}

fn path_from_relative(root: &KaosPath, relative_path: &str) -> KaosPath {
    relative_path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .fold(root.clone(), |path, segment| path.joinpath(segment))
}

fn is_script_asset(relative_path: &str) -> bool {
    let mut components = relative_path
        .split('/')
        .filter(|segment| !segment.is_empty());
    let _skill_name = components.next();
    matches!(components.next(), Some("scripts"))
}

#[cfg(test)]
mod tests {
    use super::{
        EmbeddedFile, READY_MARKER_FILE, builtin_bundle_id, builtin_bundle_root,
        is_materialized_bundle_usable, is_script_asset, materialize_embedded_files,
        write_ready_marker,
    };
    use std::sync::Arc;

    use kaos::{
        CurrentKaosToken, ExecOptions, Kaos, KaosPath, KaosProcess, LineStream, LocalKaos,
        StrOrKaosPath, reset_current_kaos, set_current_kaos, with_current_kaos_scope,
    };
    use tempfile::TempDir;

    struct FixedHomeKaos {
        inner: LocalKaos,
        home: KaosPath,
    }

    impl FixedHomeKaos {
        fn new(home: KaosPath) -> Self {
            Self {
                inner: LocalKaos::new(),
                home,
            }
        }
    }

    #[async_trait::async_trait]
    impl Kaos for FixedHomeKaos {
        fn name(&self) -> &str {
            "local"
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

        fn app_state_dir(&self, app_name: &str) -> KaosPath {
            self.home().joinpath(&format!(".{app_name}"))
        }

        fn cwd(&self) -> KaosPath {
            self.inner.cwd()
        }

        async fn chdir(&self, path: &KaosPath) -> anyhow::Result<()> {
            self.inner.chdir(path).await
        }

        async fn stat(
            &self,
            path: &KaosPath,
            follow_symlinks: bool,
        ) -> anyhow::Result<kaos::StatResult> {
            self.inner.stat(path, follow_symlinks).await
        }

        async fn iterdir(&self, path: &KaosPath) -> anyhow::Result<Vec<KaosPath>> {
            self.inner.iterdir(path).await
        }

        async fn glob(
            &self,
            path: &KaosPath,
            pattern: &str,
            case_sensitive: bool,
        ) -> anyhow::Result<Vec<KaosPath>> {
            self.inner.glob(path, pattern, case_sensitive).await
        }

        async fn read_bytes(
            &self,
            path: &KaosPath,
            limit: Option<usize>,
        ) -> anyhow::Result<Vec<u8>> {
            self.inner.read_bytes(path, limit).await
        }

        async fn read_text(&self, path: &KaosPath) -> anyhow::Result<String> {
            self.inner.read_text(path).await
        }

        async fn read_lines(&self, path: &KaosPath) -> anyhow::Result<Vec<String>> {
            self.inner.read_lines(path).await
        }

        async fn read_lines_stream(&self, path: &KaosPath) -> anyhow::Result<LineStream> {
            self.inner.read_lines_stream(path).await
        }

        async fn write_bytes(&self, path: &KaosPath, data: &[u8]) -> anyhow::Result<usize> {
            self.inner.write_bytes(path, data).await
        }

        async fn write_text(
            &self,
            path: &KaosPath,
            data: &str,
            append: bool,
        ) -> anyhow::Result<usize> {
            self.inner.write_text(path, data, append).await
        }

        async fn chmod(&self, path: &KaosPath, mode: u32) -> anyhow::Result<()> {
            self.inner.chmod(path, mode).await
        }

        async fn mkdir(
            &self,
            path: &KaosPath,
            parents: bool,
            exist_ok: bool,
        ) -> anyhow::Result<()> {
            self.inner.mkdir(path, parents, exist_ok).await
        }

        async fn env_var(&self, key: &str) -> anyhow::Result<Option<String>> {
            self.inner.env_var(key).await
        }

        async fn exec(
            &self,
            args: &[String],
            _options: ExecOptions,
        ) -> anyhow::Result<Box<dyn KaosProcess>> {
            self.inner.exec(args, ExecOptions::default()).await
        }
    }

    struct FixedHomeKaosGuard {
        token: Option<CurrentKaosToken>,
    }

    impl FixedHomeKaosGuard {
        fn new(home: KaosPath) -> Self {
            let kaos = Arc::new(FixedHomeKaos::new(home));
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

    #[tokio::test]
    async fn test_is_script_asset_matches_scripts_subtree() {
        assert!(is_script_asset("skill/scripts/run.sh"));
        assert!(is_script_asset("skill/scripts/bin/run.py"));
        assert!(!is_script_asset("skill/references/ref.md"));
        assert!(!is_script_asset("skill/SKILL.md"));
    }

    #[tokio::test]
    async fn test_materialize_embedded_files_writes_nested_assets() {
        with_current_kaos_scope(async {
            let tmp = TempDir::new().expect("temp dir");
            let home_dir = tmp.path().join("home");
            std::fs::create_dir_all(&home_dir).expect("create home dir");
            let _guard = FixedHomeKaosGuard::new(KaosPath::unsafe_from_local_path(&home_dir));
            let root = KaosPath::home() / ".kimi" / "builtin-skills" / "test-bundle";
            let assets = [
                EmbeddedFile {
                    relative_path: "demo/SKILL.md",
                    contents: b"# demo\n",
                },
                EmbeddedFile {
                    relative_path: "demo/references/ref.md",
                    contents: b"reference\n",
                },
                EmbeddedFile {
                    relative_path: "demo/scripts/run.sh",
                    contents: b"#!/bin/sh\necho ok\n",
                },
            ];

            materialize_embedded_files(&root, &assets)
                .await
                .expect("materialize builtin files");

            assert_eq!(
                (root.clone() / "demo" / "SKILL.md")
                    .read_text()
                    .await
                    .expect("read skill"),
                "# demo\n"
            );
            assert_eq!(
                (root.clone() / "demo" / "references" / "ref.md")
                    .read_text()
                    .await
                    .expect("read ref"),
                "reference\n"
            );

            if !cfg!(windows) {
                let script_stat = (root / "demo" / "scripts" / "run.sh")
                    .stat(true)
                    .await
                    .expect("script stat");
                assert_eq!(script_stat.st_mode & 0o777, 0o755);
            }
        })
        .await;
    }

    #[tokio::test]
    async fn test_is_materialized_bundle_usable_accepts_ready_marker() {
        with_current_kaos_scope(async {
            let tmp = TempDir::new().expect("temp dir");
            let home_dir = tmp.path().join("home");
            std::fs::create_dir_all(&home_dir).expect("create home dir");
            let _guard = FixedHomeKaosGuard::new(KaosPath::unsafe_from_local_path(&home_dir));
            let bundle_id = builtin_bundle_id();
            let root = builtin_bundle_root(&bundle_id);

            root.mkdir(true, true).await.expect("create root");
            write_ready_marker(&root, &bundle_id)
                .await
                .expect("write ready marker");

            assert!(
                is_materialized_bundle_usable(&root, &bundle_id, &[] as &[EmbeddedFile])
                    .await
                    .expect("check bundle usability")
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_is_materialized_bundle_usable_validates_bundle_without_marker() {
        with_current_kaos_scope(async {
            let tmp = TempDir::new().expect("temp dir");
            let home_dir = tmp.path().join("home");
            std::fs::create_dir_all(&home_dir).expect("create home dir");
            let _guard = FixedHomeKaosGuard::new(KaosPath::unsafe_from_local_path(&home_dir));
            let root = KaosPath::home() / ".kimi" / "builtin-skills" / "test-bundle";
            let assets = [
                EmbeddedFile {
                    relative_path: "demo/SKILL.md",
                    contents: b"# demo\n",
                },
                EmbeddedFile {
                    relative_path: "demo/scripts/run.sh",
                    contents: b"#!/bin/sh\necho ok\n",
                },
            ];

            materialize_embedded_files(&root, &assets)
                .await
                .expect("materialize builtin files");

            let ready_marker = root.clone() / READY_MARKER_FILE;
            if ready_marker.exists(true).await {
                std::fs::remove_file(ready_marker.unsafe_to_local_path()).expect("remove marker");
            }

            assert!(
                is_materialized_bundle_usable(&root, "unused-bundle-id", &assets)
                    .await
                    .expect("check bundle usability")
            );
        })
        .await;
    }
}
