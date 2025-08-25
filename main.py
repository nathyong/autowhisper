from pathlib import Path
import argparse
import asyncio
import dataclasses
import logging
import os
import subprocess

import numpy
import onnx_asr
import onnx_asr.adapters


_SOCKET_PATH = "/tmp/autowhisper.sock"
_RECORDING_PATH = Path("/tmp/autowhisper.raw")

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class RecordingState:
    """Holds the state of the recording process."""

    model: onnx_asr.adapters.TextResultsAsrAdapter
    proc: asyncio.subprocess.Process | None = None

    def is_recording(self) -> bool:
        """Returns True if a recording is in progress."""
        return self.proc is not None

    async def start(self):
        """Starts a new recording."""
        if self.is_recording():
            raise RuntimeError("Recording is already in progress.")

        # Remove any existing recording file
        if _RECORDING_PATH.exists():
            _RECORDING_PATH.unlink()

        command = (
            ["rec"]
            + ["-t", "raw"]  # Specify raw output format
            + ["-c", "1"]  # Mono channel
            + ["-b", "32"]  # 32-bit samples
            + ["-r", "16k"]  # 16 kHz sample rate
            + ["-e", "floating-point"]  # Floating point encoding
            + [_RECORDING_PATH]
        )
        self.proc = await asyncio.create_subprocess_exec(*command)
        if self.proc:
            _log.info(f"Started recording with PID {self.proc.pid}")

    async def stop(self) -> str:
        """Stops the current recording and returns the transcribed text."""
        if not self.proc:
            raise RuntimeError("No recording is in progress.")

        # Wait one second to flush any remaining audio data
        await asyncio.sleep(0.5)

        # Terminate the process and read the audio data from stdout
        self.proc.terminate()
        _log.info(f"Stopped recording (PID {self.proc.pid})")

        # Convert the raw audio data to a NumPy array
        audio_data = numpy.frombuffer(_RECORDING_PATH.read_bytes(), dtype=numpy.float32)

        # Transcribe the audio
        _log.info("Transcribing audio...")
        result = self.model.recognize(audio_data)
        _log.info(f"Transcription: {result}")

        # Copy the transcribed text to the clipboard
        subprocess.run(
            ["xclip", "-selection", "clipboard", "-i", "-t", "TEXT"],
            input=result.encode(),
            check=True,
        )

        # Send a notification with notify-send
        subprocess.check_call(["notify-send", "autowhisper", result])

        # Clean up
        self.proc = None
        _RECORDING_PATH.unlink()

        return result


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


async def run_server():
    """Starts the Unix domain socket server."""
    assert not os.path.exists(_SOCKET_PATH)

    _log.info(f"Starting server on {_SOCKET_PATH}")

    state = RecordingState(onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2"))
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
            if os.path.exists(_SOCKET_PATH):
                os.remove(_SOCKET_PATH)
            _log.info("Socket file removed.")


async def send_startstop():
    """Connects to the server and starts/stops recording."""
    try:
        reader, writer = await asyncio.open_unix_connection(_SOCKET_PATH)
    except (ConnectionRefusedError, FileNotFoundError):
        _log.error("Could not connect to the server. Is it running?")
        return

    writer.write(b"STARTSTOP\n")
    await writer.drain()

    data = await asyncio.wait_for(reader.readline(), timeout=5)
    _log.info(f"Server response: {data.decode().strip()}")

    writer.close()
    await writer.wait_closed()


def main():
    """Parses command-line arguments and runs the appropriate mode."""
    parser = argparse.ArgumentParser(description="IPC example with asyncio Unix sockets.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--server", action="store_true", help="Run in server mode.")
    group.add_argument("--startstop", action="store_true", help="Start or stop recording.")

    args = parser.parse_args()

    if args.server:
        asyncio.run(run_server())
    elif args.startstop:
        asyncio.run(send_startstop())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
