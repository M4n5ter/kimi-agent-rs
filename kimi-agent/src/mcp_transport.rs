use std::io;
use std::time::Duration;

use kaos::{AsyncReadable, KaosProcess};
use rmcp::RoleClient;
use rmcp::transport::Transport;
use rmcp::transport::async_rw::AsyncRwTransport;
use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream};
use tokio::task::JoinHandle;

const IO_BUFFER_SIZE: usize = 8 * 1024;
const PIPE_BUFFER_SIZE: usize = 64 * 1024;
const CHILD_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

pub struct KaosChildProcessTransport {
    transport: AsyncRwTransport<RoleClient, DuplexStream, DuplexStream>,
    process_task: Option<JoinHandle<io::Result<()>>>,
    stdout_task: Option<JoinHandle<io::Result<()>>>,
    stderr_task: Option<JoinHandle<io::Result<()>>>,
}

impl KaosChildProcessTransport {
    pub fn new(mut process: Box<dyn KaosProcess>) -> io::Result<Self> {
        let stdout = process
            .take_stdout()
            .ok_or_else(|| io::Error::other("missing stdout stream"))?;
        let stderr = process
            .take_stderr()
            .ok_or_else(|| io::Error::other("missing stderr stream"))?;

        let (transport_stdin, process_stdin) = tokio::io::duplex(PIPE_BUFFER_SIZE);
        let (process_stdout, transport_stdout) = tokio::io::duplex(PIPE_BUFFER_SIZE);

        let process_task =
            tokio::spawn(async move { manage_process_lifecycle(process, process_stdin).await });
        let stdout_task =
            tokio::spawn(async move { pipe_readable_into_stream(stdout, process_stdout).await });
        let stderr_task = tokio::spawn(async move { drain_readable(stderr).await });

        Ok(Self {
            transport: AsyncRwTransport::new(transport_stdout, transport_stdin),
            process_task: Some(process_task),
            stdout_task: Some(stdout_task),
            stderr_task: Some(stderr_task),
        })
    }
}

impl Transport<RoleClient> for KaosChildProcessTransport {
    type Error = io::Error;

    fn send(
        &mut self,
        item: rmcp::service::TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        self.transport.send(item)
    }

    fn receive(
        &mut self,
    ) -> impl Future<Output = Option<rmcp::service::RxJsonRpcMessage<RoleClient>>> + Send {
        self.transport.receive()
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.transport.close().await?;
        await_task(&mut self.stdout_task, "stdout forwarder").await?;
        await_task(&mut self.stderr_task, "stderr forwarder").await?;
        await_task(&mut self.process_task, "process manager").await?;
        Ok(())
    }
}

async fn manage_process_lifecycle(
    mut process: Box<dyn KaosProcess>,
    mut stdin_source: DuplexStream,
) -> io::Result<()> {
    let mut buffer = vec![0u8; IO_BUFFER_SIZE];

    loop {
        let read = stdin_source.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        process
            .stdin()
            .write(&buffer[..read])
            .await
            .map_err(io_other)?;
        process.stdin().flush().await.map_err(io_other)?;
    }

    process.stdin().close().await.map_err(io_other)?;

    match tokio::time::timeout(CHILD_SHUTDOWN_TIMEOUT, process.wait()).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(io_other(err)),
        Err(_) => {
            process.kill().await.map_err(io_other)?;
            let _ = process.wait().await.map_err(io_other)?;
            Ok(())
        }
    }
}

async fn pipe_readable_into_stream(
    mut readable: Box<dyn AsyncReadable>,
    mut writer: DuplexStream,
) -> io::Result<()> {
    loop {
        let chunk = readable.read(IO_BUFFER_SIZE).await.map_err(io_other)?;
        if chunk.is_empty() {
            writer.shutdown().await?;
            return Ok(());
        }
        writer.write_all(&chunk).await?;
    }
}

async fn drain_readable(mut readable: Box<dyn AsyncReadable>) -> io::Result<()> {
    loop {
        let chunk = readable.read(IO_BUFFER_SIZE).await.map_err(io_other)?;
        if chunk.is_empty() {
            return Ok(());
        }
    }
}

async fn await_task(task: &mut Option<JoinHandle<io::Result<()>>>, label: &str) -> io::Result<()> {
    let Some(task) = task.take() else {
        return Ok(());
    };
    task.await
        .map_err(|err| io::Error::other(format!("{label} join failed: {err}")))?
}

fn io_other(err: impl std::fmt::Display) -> io::Error {
    io::Error::other(err.to_string())
}
