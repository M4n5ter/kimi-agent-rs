use std::collections::VecDeque;

use futures::stream;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};

use crate::LineStream;

pub(crate) fn line_stream_from_async_read<R>(reader: R) -> LineStream
where
    R: AsyncRead + Unpin + Send + 'static,
{
    struct LineStreamState<R>
    where
        R: AsyncRead + Unpin + Send,
    {
        reader: BufReader<R>,
        buf: Vec<u8>,
        pending: VecDeque<String>,
        done: bool,
    }

    let state = LineStreamState {
        reader: BufReader::new(reader),
        buf: Vec::new(),
        pending: VecDeque::new(),
        done: false,
    };

    let stream = stream::unfold(state, |mut state| async move {
        loop {
            if let Some(line) = state.pending.pop_front() {
                return Some((Ok(line), state));
            }
            if state.done {
                return None;
            }

            state.buf.clear();
            match state.reader.read_until(b'\n', &mut state.buf).await {
                Ok(0) => {
                    state.done = true;
                }
                Ok(_) => {
                    let text = String::from_utf8_lossy(&state.buf);
                    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                    for line in normalized.split_inclusive('\n').map(str::to_string) {
                        state.pending.push_back(line);
                    }
                }
                Err(err) => return Some((Err(err.into()), state)),
            }
        }
    });

    Box::pin(stream)
}
