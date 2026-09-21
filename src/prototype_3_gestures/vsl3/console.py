from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Make Vietnamese labels printable on Windows consoles and redirected streams."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # Test capture streams and already-closed streams may refuse reconfiguration.
            pass
