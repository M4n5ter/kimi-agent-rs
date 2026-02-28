# Skill Module Notes

## Scope

- `skill/mod.rs`: skill discovery, frontmatter parsing, layered roots.
- `skill/flow/*`: flow parsing for mermaid and d2.
- `utils/frontmatter.rs`: YAML frontmatter extraction.

## Compatibility Rules

- Skills are discovered from layered roots (builtin → user → project) with later roots overriding.
- Builtin skills are embedded into the binary and synchronized into a managed directory under the active Kaos backend app state dir before discovery runs.
- User/project skills remain Kaos-backed directories discovered from the active backend filesystem.
- Builtin sync is best-effort: if the managed directory cannot be prepared, discovery skips builtin skills and continues.
- Flow skills require a `mermaid` or `d2` fenced code block in `SKILL.md`.
- Invalid flow parsing falls back to `standard` skill type.
- Skill frontmatter may include `mcp` server definitions (`stdio`/`http`) for runtime dynamic loading.
