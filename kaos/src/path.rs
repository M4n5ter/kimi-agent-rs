use std::fmt;
use std::hash::{Hash, Hasher};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use typed_path::{PathType, Utf8TypedPath, Utf8TypedPathBuf};

use crate::{
    LineStream, StatResult, StrOrKaosPath, get_current_kaos, normalize_path_arg, normpath,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KaosPathStyle {
    Posix,
    Windows,
}

impl KaosPathStyle {
    pub fn local_default() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else {
            Self::Posix
        }
    }

    fn path_type(self) -> PathType {
        match self {
            Self::Posix => PathType::Unix,
            Self::Windows => PathType::Windows,
        }
    }

    fn separator(self) -> &'static str {
        match self {
            Self::Posix => "/",
            Self::Windows => "\\",
        }
    }
}

#[derive(Clone)]
pub struct KaosPath {
    style: KaosPathStyle,
    raw: String,
    local_bytes: Option<Vec<u8>>,
}

impl KaosPath {
    pub fn new(path: impl AsRef<str>) -> Self {
        Self {
            style: get_current_kaos().path_style(),
            raw: path.as_ref().to_string(),
            local_bytes: None,
        }
    }

    pub fn from_style(style: KaosPathStyle, path: impl AsRef<str>) -> Self {
        Self {
            style,
            raw: path.as_ref().to_string(),
            local_bytes: None,
        }
    }

    pub fn from(path: PathBuf) -> Self {
        Self::from_local_pathbuf(path)
    }

    pub fn from_local_pathbuf(path: PathBuf) -> Self {
        #[cfg(unix)]
        {
            let bytes = path.as_os_str().as_bytes().to_vec();
            Self {
                style: KaosPathStyle::local_default(),
                raw: path.to_string_lossy().to_string(),
                local_bytes: Some(bytes),
            }
        }

        #[cfg(not(unix))]
        {
            Self {
                style: KaosPathStyle::local_default(),
                raw: path.to_string_lossy().to_string(),
                local_bytes: None,
            }
        }
    }

    pub fn style(&self) -> KaosPathStyle {
        self.style
    }

    pub fn unsafe_from_local_path(path: &Path) -> Self {
        Self::from_local_pathbuf(path.to_path_buf())
    }

    pub fn unsafe_to_local_path(&self) -> PathBuf {
        #[cfg(unix)]
        if let Some(bytes) = &self.local_bytes {
            return PathBuf::from(std::ffi::OsString::from_vec(bytes.clone()));
        }
        PathBuf::from(&self.raw)
    }

    pub fn name(&self) -> String {
        self.as_typed_path()
            .file_name()
            .map(str::to_string)
            .unwrap_or_default()
    }

    pub fn parent(&self) -> KaosPath {
        if let Some(parent) = self.as_typed_path().parent() {
            return Self::from_style(self.style, parent.as_str());
        }
        if self.is_absolute() {
            return self.clone();
        }
        Self::from_style(self.style, ".")
    }

    pub fn is_absolute(&self) -> bool {
        self.as_typed_path().is_absolute()
    }

    pub fn joinpath(&self, other: &str) -> Self {
        Self::from_typed_path_buf(self.as_typed_path().join(other))
    }

    pub fn canonical(&self) -> KaosPath {
        let abs = if self.is_absolute() {
            self.clone()
        } else {
            let cwd = get_current_kaos().cwd();
            cwd.joinpath(&self.raw)
        };
        normpath(&StrOrKaosPath::KaosPath(&abs))
    }

    pub fn relative_to(&self, other: &KaosPath) -> Result<KaosPath> {
        if self.style != other.style {
            return Err(anyhow!(
                "Cannot compare paths with different styles: {:?} vs {:?}",
                self.style,
                other.style
            ));
        }
        let this = self.as_typed_path();
        let relative = this
            .strip_prefix(other.as_typed_path().as_str())
            .map_err(|err| anyhow!(err))?;
        Ok(Self::from_style(self.style, relative.as_str()))
    }

    pub fn home() -> KaosPath {
        get_current_kaos().home()
    }

    pub fn cwd() -> KaosPath {
        get_current_kaos().cwd()
    }

    pub fn expanduser(&self) -> KaosPath {
        if self.raw == "~" {
            return KaosPath::home();
        }

        let home = KaosPath::home();
        let posix_prefix = "~/";
        let windows_prefix = "~\\";

        if self.raw.starts_with(posix_prefix) {
            return home.joinpath(&self.raw[posix_prefix.len()..]);
        }
        if self.raw.starts_with(windows_prefix) {
            return home.joinpath(&self.raw[windows_prefix.len()..]);
        }

        self.clone()
    }

    pub async fn stat(&self, follow_symlinks: bool) -> Result<StatResult> {
        get_current_kaos().stat(self, follow_symlinks).await
    }

    pub async fn exists(&self, follow_symlinks: bool) -> bool {
        self.stat(follow_symlinks).await.is_ok()
    }

    pub async fn is_file(&self, follow_symlinks: bool) -> bool {
        match self.stat(follow_symlinks).await {
            Ok(stat) => mode_is_file(stat.st_mode),
            Err(_) => false,
        }
    }

    pub async fn is_dir(&self, follow_symlinks: bool) -> bool {
        match self.stat(follow_symlinks).await {
            Ok(stat) => mode_is_dir(stat.st_mode),
            Err(_) => false,
        }
    }

    pub async fn iterdir(&self) -> Result<Vec<KaosPath>> {
        get_current_kaos().iterdir(self).await
    }

    pub async fn glob(&self, pattern: &str, case_sensitive: bool) -> Result<Vec<KaosPath>> {
        get_current_kaos().glob(self, pattern, case_sensitive).await
    }

    pub async fn read_bytes(&self, limit: Option<usize>) -> Result<Vec<u8>> {
        get_current_kaos().read_bytes(self, limit).await
    }

    pub async fn read_text(&self) -> Result<String> {
        get_current_kaos().read_text(self).await
    }

    pub async fn read_lines(&self) -> Result<Vec<String>> {
        get_current_kaos().read_lines(self).await
    }

    pub async fn read_lines_stream(&self) -> Result<LineStream> {
        get_current_kaos().read_lines_stream(self).await
    }

    pub async fn write_bytes(&self, data: &[u8]) -> Result<usize> {
        get_current_kaos().write_bytes(self, data).await
    }

    pub async fn write_text(&self, data: &str) -> Result<usize> {
        get_current_kaos().write_text(self, data, false).await
    }

    pub async fn append_text(&self, data: &str) -> Result<usize> {
        get_current_kaos().write_text(self, data, true).await
    }

    pub async fn chmod(&self, mode: u32) -> Result<()> {
        get_current_kaos().chmod(self, mode).await
    }

    pub async fn mkdir(&self, parents: bool, exist_ok: bool) -> Result<()> {
        get_current_kaos().mkdir(self, parents, exist_ok).await
    }

    pub fn to_string_lossy(&self) -> String {
        self.raw.clone()
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn separator(&self) -> &'static str {
        self.style.separator()
    }

    fn as_typed_path(&self) -> Utf8TypedPath<'_> {
        Utf8TypedPath::new(&self.raw, self.style.path_type())
    }

    fn from_typed_path_buf(path: Utf8TypedPathBuf) -> Self {
        match path {
            Utf8TypedPathBuf::Unix(p) => Self::from_style(KaosPathStyle::Posix, p.as_str()),
            Utf8TypedPathBuf::Windows(p) => Self::from_style(KaosPathStyle::Windows, p.as_str()),
        }
    }
}

impl fmt::Debug for KaosPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KaosPath")
            .field("style", &self.style)
            .field("raw", &self.raw)
            .finish()
    }
}

impl PartialEq for KaosPath {
    fn eq(&self, other: &Self) -> bool {
        self.style == other.style && self.raw == other.raw
    }
}

impl Eq for KaosPath {}

impl Hash for KaosPath {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.style.hash(state);
        self.raw.hash(state);
    }
}

impl fmt::Display for KaosPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

impl From<&str> for KaosPath {
    fn from(value: &str) -> Self {
        KaosPath::new(value)
    }
}

impl From<PathBuf> for KaosPath {
    fn from(value: PathBuf) -> Self {
        KaosPath::from_local_pathbuf(value)
    }
}

impl From<&Path> for KaosPath {
    fn from(value: &Path) -> Self {
        KaosPath::from_local_pathbuf(value.to_path_buf())
    }
}

impl std::ops::Div<&str> for KaosPath {
    type Output = KaosPath;

    fn div(self, rhs: &str) -> Self::Output {
        self.joinpath(rhs)
    }
}

impl std::ops::Div<&KaosPath> for KaosPath {
    type Output = KaosPath;

    fn div(self, rhs: &KaosPath) -> Self::Output {
        self.joinpath(&rhs.to_string_lossy())
    }
}

pub fn normalize_path(arg: &crate::StrOrKaosPath<'_>) -> KaosPath {
    let path = normalize_path_arg(arg);
    KaosPath::from_typed_path_buf(path.as_typed_path().normalize())
}

fn mode_is_dir(mode: u32) -> bool {
    (mode & 0o170000) == 0o040000
}

fn mode_is_file(mode: u32) -> bool {
    (mode & 0o170000) == 0o100000
}

#[cfg(test)]
mod tests {
    use super::{KaosPath, KaosPathStyle};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::path::PathBuf;

    #[cfg(unix)]
    #[test]
    fn from_local_pathbuf_preserves_non_utf8_bytes() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let raw = b"/tmp/kimi-\xFF".to_vec();
        let path = PathBuf::from(std::ffi::OsString::from_vec(raw.clone()));
        let kaos_path = KaosPath::from_local_pathbuf(path);
        let roundtrip = kaos_path.unsafe_to_local_path();

        assert_eq!(roundtrip.as_os_str().as_bytes(), raw.as_slice());
    }

    #[cfg(unix)]
    #[test]
    fn equality_and_hash_ignore_local_bytes_cache() {
        let local = KaosPath::from_local_pathbuf(PathBuf::from("/tmp/kimi-path"));
        let logical = KaosPath::from_style(KaosPathStyle::Posix, "/tmp/kimi-path");

        assert_eq!(local, logical);

        let mut left = DefaultHasher::new();
        local.hash(&mut left);
        let mut right = DefaultHasher::new();
        logical.hash(&mut right);
        assert_eq!(left.finish(), right.finish());
    }
}
