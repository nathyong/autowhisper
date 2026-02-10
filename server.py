# Server section of autowhisper
# Split such that main.py has no dependencies if run as a client

import asyncio
import concurrent.futures as cf
import dataclasses
import logging
import os
from pathlib import Path
import subprocess
from typing import NamedTuple

import wave

import numpy
import onnx_asr
import onnx_asr.adapters

_SOCKET_PATH = "/tmp/autowhisper.sock"
_RECORDING_PATH = Path("/tmp/autowhisper.raw")
_DEBUG_WAV_PATH = Path("/tmp/autowhisper_debug.wav")

_log = logging.getLogger(__name__)


class Recording(NamedTuple):
    """Represents a recording queued for transcription."""

    audio_data: numpy.ndarray


@dataclasses.dataclass
class RecordingState:
    """Holds the state of the recording process."""

    transcription_queue: asyncio.Queue[Recording]
    _lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    proc: asyncio.subprocess.Process | None = None

    def is_recording(self) -> bool:
        """Returns True if a recording is in progress."""
        return self.proc is not None

    async def start(self):
        """Starts a new recording."""
        async with self._lock:
            if self.is_recording():
                raise RuntimeError("Recording is already in progress.")

            # Remove any existing recording file
            if _RECORDING_PATH.exists():
                _RECORDING_PATH.unlink()

            command = [
                "pw-record",
                "--raw",              # RAW mode (no container)
                "--format", "f32",    # 32-bit float
                "--rate", "16000",    # 16 kHz sample rate
                "--channels", "1",    # Mono
                "--latency", "10ms",  # Small buffer to minimise loss on stop
                "-",                  # Write to stdout
            ]
            outfile = open(_RECORDING_PATH, "wb")
            self.proc = await asyncio.create_subprocess_exec(
                *command, stdout=outfile
            )
            outfile.close()
            if self.proc:
                _log.info(f"Started recording with PID {self.proc.pid}")

    async def stop(self) -> None:
        """Stops the current recording and queues it for transcription."""
        async with self._lock:
            if not self.proc:
                raise RuntimeError("No recording is in progress.")

            # Terminate the process and wait for it to exit
            self.proc.terminate()
            await self.proc.wait()
            _log.info(f"Stopped recording (PID {self.proc.pid})")

            try:
                # Convert the raw audio data to a NumPy array
                audio_data = numpy.frombuffer(
                    _RECORDING_PATH.read_bytes(), dtype=numpy.float32
                )

                # Export a WAV copy for debugging
                pcm16 = (audio_data * 32767).clip(-32768, 32767).astype(numpy.int16)
                with wave.open(str(_DEBUG_WAV_PATH), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(pcm16.tobytes())
                _log.info(f"Debug WAV written to {_DEBUG_WAV_PATH}")

                # Skip if the audio data is too short (less than 1s at 16kHz)
                if len(audio_data) < 16000:
                    _log.info("Audio data is too short, skipping transcription.")
                    return

                # Skip if the audio data is too long (more than 100s at 16kHz)
                if len(audio_data) > 100 * 16000:
                    _log.info("Audio data is too long, skipping transcription.")
                    return

                if numpy.all(audio_data == 0):
                    subprocess.check_call(
                        [
                            "notify-send",
                            "autowhisper",
                            "[[Audio data is silent, skipping transcription.]]",
                        ]
                    )
                    _log.info("Audio data is silent, skipping transcription.")
                    return

                # Queue the audio for transcription
                await self.transcription_queue.put(Recording(audio_data))
                _log.info("Audio queued for transcription")

            finally:
                # Clean up
                self.proc = None
                # _RECORDING_PATH.unlink(missing_ok=True)


async def handle_command(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, state: RecordingState
):
    """Callback function for when a client connects to the server."""
    message = (await reader.read(100)).strip()
    if message == b"STARTSTOP":
        if state.is_recording():
            await state.stop()
            writer.write(b"STOPPED\n")
        else:
            await state.start()
            writer.write(b"STARTED\n")
        await writer.drain()
    else:
        _log.warning(f"Received unknown command: {message}")
        writer.write(b"UNKNOWN\n")
        await writer.drain()

    _log.info("Closing connection")
    writer.close()
    await writer.wait_closed()


async def transcription_worker(
    model: onnx_asr.adapters.TextResultsAsrAdapter,
    queue: asyncio.Queue[Recording],
    executor: cf.ThreadPoolExecutor,
):
    """Background worker that processes transcription requests from the queue."""
    _log.info("Transcription worker started")

    while True:
        try:
            # Wait for a recording to transcribe
            queued_recording = await queue.get()
            audio_data = queued_recording.audio_data

            _log.info(f"Processing transcription (queue size: {queue.qsize()})")

            # Transcribe the audio in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, model.recognize, audio_data)

            if not result:
                subprocess.check_call(
                    [
                        "notify-send",
                        "autowhisper",
                        "--urgency=critical",
                        "Transcription failed or returned no result.",
                    ]
                )
                _log.info("Transcription failed or returned no result.")
            else:
                _log.info(f"Transcription: {result}")

                # Copy the transcribed text to the clipboard and notify user
                subprocess.run(
                    ["xsel", "--clipboard", "-i"],
                    input=result.encode(),
                    check=True,
                )
                subprocess.check_call(["notify-send", "autowhisper", result])

            # Mark the task as done
            queue.task_done()

        except asyncio.CancelledError:
            _log.info("Transcription worker shutting down")
            break
        except Exception as e:
            _log.error(f"Error in transcription worker: {e}")
            queue.task_done()


async def run_server():
    """Starts the Unix domain socket server."""
    if os.path.exists(_SOCKET_PATH):
        # Check if another server is already listening
        try:
            reader, writer = await asyncio.open_unix_connection(_SOCKET_PATH)
            writer.close()
            await writer.wait_closed()
            raise RuntimeError(f"Another server is already running on {_SOCKET_PATH}")
        except (ConnectionRefusedError, FileNotFoundError):
            _log.warning(f"Removing stale socket file: {_SOCKET_PATH}")
            os.remove(_SOCKET_PATH)

    _log.info("Loading model")
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")

    # Create transcription queue and executor
    transcription_queue = asyncio.Queue[Recording]()
    executor = cf.ThreadPoolExecutor(max_workers=1)

    # Create recording state
    state = RecordingState(transcription_queue)

    # Start the transcription worker
    worker_task = asyncio.create_task(
        transcription_worker(model, transcription_queue, executor)
    )

    _log.info(f"Starting server on {_SOCKET_PATH}")
    server = await asyncio.start_unix_server(
        lambda r, w: handle_command(r, w, state), path=_SOCKET_PATH
    )

    async with server:
        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            _log.info("Server shutting down.")
            if state.is_recording():
                await state.stop()
        finally:
            # Clean up worker and executor
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

            executor.shutdown(wait=True)
            if os.path.exists(_SOCKET_PATH):
                os.remove(_SOCKET_PATH)
            _log.info("Socket file removed.")
