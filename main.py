import argparse
import asyncio

import logger

_SOCKET_PATH = "/tmp/autowhisper.sock"

_log = logger.get_logger(__name__)


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
    _log.info("Server response", response=data.decode().strip())

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
        import server

        asyncio.run(server.run_server())
    elif args.startstop:
        asyncio.run(send_startstop())


if __name__ == "__main__":
    main()
