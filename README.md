# autowhisper

Push-to-talk speech-to-text for Linux. Despite the name, Whisper is not involved.

- `main.py --startstop`: sends commands over a Unix socket.
- `server.py`: recording and transcription with [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model via ONNX Runtime. Recordings are capped at 100 seconds. 
    - Uses `pw-record` (pipewire) for recording
    - Silent/short recordings are discarded
    - 100s timeout
    - Transcriptions are sent to the clipboard and a notification

Usage:
- `just setup`
- `just run`
- bind `python main.py --startstop` to a hotkey

Dependencies: PipeWire, `uv`, `xsel`, `libnotify`.

