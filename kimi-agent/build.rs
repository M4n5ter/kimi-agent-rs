use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let skills_dir = manifest_dir.join("src").join("skills");
    println!("cargo:rerun-if-changed={}", skills_dir.display());

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let dest = out_dir.join("builtin_skills.rs");
    let files = collect_files(&skills_dir, &skills_dir);
    let generated = render_builtin_assets(&files);
    fs::write(&dest, generated).expect("write builtin skills manifest");
}

fn collect_files(root: &Path, dir: &Path) -> Vec<(String, PathBuf)> {
    let mut entries = Vec::new();
    let mut children: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("read_dir {}: {err}", dir.display()))
        .map(|entry| entry.unwrap_or_else(|err| panic!("read_dir entry {}: {err}", dir.display())))
        .collect();
    children.sort_by_key(|entry| entry.path());

    for child in children {
        let path = child.path();
        if child
            .file_type()
            .unwrap_or_else(|err| panic!("file_type {}: {err}", path.display()))
            .is_dir()
        {
            entries.extend(collect_files(root, &path));
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .unwrap_or_else(|err| panic!("strip_prefix {}: {err}", path.display()))
            .to_string_lossy()
            .replace('\\', "/");
        entries.push((relative, path));
    }

    entries
}

fn render_builtin_assets(files: &[(String, PathBuf)]) -> String {
    let mut generated = String::from("static BUILTIN_SKILL_FILES: &[EmbeddedFile] = &[\n");
    for (relative, absolute) in files {
        generated.push_str("    EmbeddedFile {\n");
        generated.push_str(&format!("        relative_path: {relative:?},\n"));
        generated.push_str(&format!(
            "        contents: include_bytes!(r#\"{}\"#),\n",
            absolute.display()
        ));
        generated.push_str("    },\n");
    }
    generated.push_str("];\n");
    generated
}
