use std::io;
use std::sync::Arc;

use kaos::Kaos;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use url::Url;

const MAX_HEADER_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub struct KaosHttpProxyHandle {
    listen_addr: std::net::SocketAddr,
    shutdown: CancellationToken,
    task: Option<JoinHandle<io::Result<()>>>,
}

impl KaosHttpProxyHandle {
    pub async fn bind(kaos: Arc<dyn Kaos>) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let listen_addr = listener.local_addr()?;
        let shutdown = CancellationToken::new();
        let task_shutdown = shutdown.clone();
        let task = tokio::spawn(async move { run_proxy(listener, kaos, task_shutdown).await });

        Ok(Self {
            listen_addr,
            shutdown,
            task: Some(task),
        })
    }

    pub fn proxy_url(&self) -> String {
        format!("http://{}", self.listen_addr)
    }

    pub async fn close(&mut self) -> io::Result<()> {
        self.shutdown.cancel();
        if let Some(task) = self.task.take() {
            task.await
                .map_err(|err| io::Error::other(format!("proxy task join failed: {err}")))??;
        }
        Ok(())
    }
}

impl Drop for KaosHttpProxyHandle {
    fn drop(&mut self) {
        // Drop cannot await the proxy task, but it must still stop accepting
        // new work on early-return paths that never reach `close()`.
        self.shutdown.cancel();
    }
}

async fn run_proxy(
    listener: TcpListener,
    kaos: Arc<dyn Kaos>,
    shutdown: CancellationToken,
) -> io::Result<()> {
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            accepted = listener.accept() => {
                let (socket, _) = accepted?;
                let connection_kaos = Arc::clone(&kaos);
                tokio::spawn(async move {
                    if let Err(err) = handle_connection(socket, connection_kaos).await {
                        tracing::debug!("mcp proxy connection failed: {err}");
                    }
                });
            }
        }
    }
}

async fn handle_connection(mut client: TcpStream, kaos: Arc<dyn Kaos>) -> io::Result<()> {
    let (head, buffered_body) = read_http_head(&mut client).await?;
    let request = ParsedHttpRequest::parse(&head)?;

    match request {
        ParsedHttpRequest::Connect { authority } => {
            let (host, port) = split_authority(&authority)?;
            let mut upstream = kaos.connect_tcp(host, port).await.map_err(io_other)?;
            client
                .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                .await?;
            let _ = tokio::io::copy_bidirectional(&mut client, &mut *upstream).await?;
            Ok(())
        }
        ParsedHttpRequest::Forward(forward) => {
            let mut upstream = kaos
                .connect_tcp(&forward.host, forward.port)
                .await
                .map_err(io_other)?;
            upstream
                .write_all(forward.rewritten_head.as_bytes())
                .await?;
            if !buffered_body.is_empty() {
                upstream.write_all(&buffered_body).await?;
            }
            let _ = tokio::io::copy_bidirectional(&mut client, &mut *upstream).await?;
            Ok(())
        }
    }
}

async fn read_http_head(stream: &mut TcpStream) -> io::Result<(Vec<u8>, Vec<u8>)> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];

    loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading proxy request",
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(idx) = find_header_end(&buffer) {
            let body = buffer.split_off(idx + 4);
            return Ok((buffer, body));
        }
        if buffer.len() > MAX_HEADER_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "proxy request headers exceed limit",
            ));
        }
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

enum ParsedHttpRequest {
    Connect { authority: String },
    Forward(ForwardRequest),
}

struct ForwardRequest {
    host: String,
    port: u16,
    rewritten_head: String,
}

impl ParsedHttpRequest {
    fn parse(head: &[u8]) -> io::Result<Self> {
        let head_str = std::str::from_utf8(head)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        let mut lines = head_str.split("\r\n");
        let request_line = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?;
        let mut request_parts = request_line.splitn(3, ' ');
        let method = request_parts
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing method"))?;
        let target = request_parts
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing target"))?;
        let version = request_parts
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing version"))?;

        let headers = lines
            .take_while(|line| !line.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();

        if method.eq_ignore_ascii_case("CONNECT") {
            return Ok(Self::Connect {
                authority: target.to_string(),
            });
        }

        let (host, port, path_and_query) = rewrite_target(target, &headers)?;
        let mut rewritten_head = format!("{method} {path_and_query} {version}\r\n");
        for header in headers {
            let Some((name, _value)) = header.split_once(':') else {
                continue;
            };
            if name.eq_ignore_ascii_case("connection")
                || name.eq_ignore_ascii_case("proxy-connection")
                || name.eq_ignore_ascii_case("proxy-authorization")
            {
                continue;
            }
            rewritten_head.push_str(&header);
            rewritten_head.push_str("\r\n");
        }
        rewritten_head.push_str("Connection: close\r\n\r\n");

        Ok(Self::Forward(ForwardRequest {
            host,
            port,
            rewritten_head,
        }))
    }
}

fn rewrite_target(target: &str, headers: &[String]) -> io::Result<(String, u16, String)> {
    let url = if target.starts_with("http://") || target.starts_with("https://") {
        Url::parse(target).map_err(io_other)?
    } else {
        let host = headers
            .iter()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("host")
                    .then(|| value.trim().to_string())
            })
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing Host header"))?;
        Url::parse(&format!("http://{host}{target}")).map_err(io_other)?
    };

    let host = url
        .host_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "proxy target missing host"))?
        .to_string();
    let port = url.port_or_known_default().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "proxy target missing resolved port",
        )
    })?;

    let mut path_and_query = url.path().to_string();
    if let Some(query) = url.query() {
        path_and_query.push('?');
        path_and_query.push_str(query);
    }
    if path_and_query.is_empty() {
        path_and_query.push('/');
    }

    Ok((host, port, path_and_query))
}

fn split_authority(authority: &str) -> io::Result<(&str, u16)> {
    let mut parts = authority.rsplitn(2, ':');
    let port = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing proxy port"))?;
    let host = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing proxy host"))?;
    let port = port.parse::<u16>().map_err(io_other)?;
    Ok((host.trim_matches(['[', ']']), port))
}

fn io_other(err: impl std::fmt::Display) -> io::Error {
    io::Error::other(err.to_string())
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;
    use std::sync::Arc;
    use std::time::Duration;

    use kaos::LocalKaos;
    use tokio::net::TcpStream;
    use tokio::time::{Instant, sleep};

    use super::{KaosHttpProxyHandle, ParsedHttpRequest, rewrite_target, split_authority};

    #[test]
    fn split_authority_supports_ipv4_and_ipv6() {
        let (host, port) = split_authority("example.com:443").expect("split ipv4");
        assert_eq!(host, "example.com");
        assert_eq!(port, 443);

        let (host, port) = split_authority("[::1]:8443").expect("split ipv6");
        assert_eq!(host, "::1");
        assert_eq!(port, 8443);
    }

    #[test]
    fn rewrite_target_handles_absolute_and_origin_form() {
        let headers = vec!["Host: api.example.com".to_string()];

        let (host, port, path) =
            rewrite_target("https://api.example.com/v1/mcp?stream=1", &headers)
                .expect("rewrite absolute target");
        assert_eq!(host, "api.example.com");
        assert_eq!(port, 443);
        assert_eq!(path, "/v1/mcp?stream=1");

        let (host, port, path) =
            rewrite_target("/oauth/token", &headers).expect("rewrite origin target");
        assert_eq!(host, "api.example.com");
        assert_eq!(port, 80);
        assert_eq!(path, "/oauth/token");
    }

    #[test]
    fn parse_forward_request_rewrites_proxy_headers() {
        let request = b"POST https://api.example.com/oauth/token HTTP/1.1\r\nHost: api.example.com\r\nProxy-Connection: keep-alive\r\nConnection: keep-alive\r\nContent-Type: application/json\r\n\r\n";
        let parsed = ParsedHttpRequest::parse(request).expect("parse request");

        let ParsedHttpRequest::Forward(forward) = parsed else {
            panic!("expected forward request");
        };
        assert_eq!(forward.host, "api.example.com");
        assert_eq!(forward.port, 443);
        assert!(
            forward
                .rewritten_head
                .starts_with("POST /oauth/token HTTP/1.1\r\n")
        );
        assert!(forward.rewritten_head.contains("Host: api.example.com\r\n"));
        assert!(
            forward
                .rewritten_head
                .contains("Content-Type: application/json\r\n")
        );
        assert!(!forward.rewritten_head.contains("Proxy-Connection"));
        assert!(!forward.rewritten_head.contains("Connection: keep-alive"));
        assert!(
            forward
                .rewritten_head
                .ends_with("Connection: close\r\n\r\n")
        );
    }

    #[tokio::test]
    async fn dropping_proxy_handle_stops_accepting_connections() {
        let proxy_url = {
            let proxy = KaosHttpProxyHandle::bind(Arc::new(LocalKaos::new()))
                .await
                .expect("bind proxy");
            let proxy_url = proxy.proxy_url();
            TcpStream::connect(proxy_url.strip_prefix("http://").expect("proxy host"))
                .await
                .expect("proxy accepts connections before drop");
            proxy_url
        };

        let proxy_host = proxy_url.strip_prefix("http://").expect("proxy host");
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            match TcpStream::connect(proxy_host).await {
                Ok(stream) => {
                    drop(stream);
                    assert!(
                        Instant::now() < deadline,
                        "proxy still accepted connections after drop"
                    );
                    sleep(Duration::from_millis(10)).await;
                }
                Err(err) => {
                    assert!(
                        matches!(
                            err.kind(),
                            ErrorKind::ConnectionRefused
                                | ErrorKind::ConnectionAborted
                                | ErrorKind::ConnectionReset
                                | ErrorKind::NotConnected
                        ),
                        "unexpected connection error after drop: {err}"
                    );
                    break;
                }
            }
        }
    }
}
