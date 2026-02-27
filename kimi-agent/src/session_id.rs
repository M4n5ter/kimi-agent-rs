use std::path::{Component, Path};

pub fn normalize_session_id(session_id: &str) -> anyhow::Result<String> {
    let normalized = session_id.trim();
    if normalized.is_empty() {
        anyhow::bail!("session_id cannot be empty");
    }
    if normalized.len() > 128 {
        anyhow::bail!("session_id is too long (max 128 chars)");
    }
    if normalized.contains('/') || normalized.contains('\\') {
        anyhow::bail!("session_id cannot contain path separators");
    }
    if normalized == "." || normalized == ".." {
        anyhow::bail!("session_id cannot be dot segments");
    }
    if !normalized
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        anyhow::bail!("session_id contains invalid characters");
    }

    let mut components = Path::new(normalized).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => {}
        _ => anyhow::bail!("session_id must be a single path segment"),
    }

    Ok(normalized.to_string())
}

#[cfg(test)]
mod tests {
    use super::normalize_session_id;

    #[test]
    fn normalize_session_id_accepts_valid_session_id() {
        assert_eq!(normalize_session_id("abc").unwrap(), "abc");
        assert_eq!(normalize_session_id(" abc-def ").unwrap(), "abc-def");
    }

    #[test]
    fn normalize_session_id_rejects_invalid_session_id() {
        assert!(normalize_session_id("").is_err());
        assert!(normalize_session_id("   ").is_err());
        assert!(normalize_session_id(".").is_err());
        assert!(normalize_session_id("..").is_err());
        assert!(normalize_session_id("a/b").is_err());
        assert!(normalize_session_id("a\\b").is_err());
        assert!(normalize_session_id("a:b").is_err());
        assert!(normalize_session_id("a b").is_err());
        assert!(normalize_session_id(&"a".repeat(129)).is_err());
    }
}
