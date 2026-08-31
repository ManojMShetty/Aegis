"""A standard-library HTTP transport over :mod:`aegis.console.api`.

Deliberately thin. Every judgement the console makes lives in ``api.py``, which
has no sockets in it and is therefore tested directly; what is left here is
routing, JSON encoding and the two safety properties a local server owes:

* **Loopback only.** The bind address is not configurable. This repository's
  security posture is that nothing here is reachable from off the host: this process
  holds no API key, builds no model client and makes no outbound request, and
  ``docker/docker-compose.yml`` puts the eval service on an ``internal: true`` network
  (which was in force for no recorded run — the agent under test is hosted). A demo that
  could be exposed on a LAN with a flag would be the one hole in it. ``127.0.0.1`` is a constant.
* **No file-reading endpoint.** Exactly one static asset is served, addressed by
  a module-level constant rather than by anything in the URL. The process runs
  with its working directory at a repository root that contains a real,
  gitignored ``.env``; a ``?path=`` parameter here would be a credential-read
  primitive one careless generalisation later.

WHY NOT FASTAPI
---------------
``pyproject.toml`` states the rule: every dependency must be imported by code in
this repository, after a trim that removed fifteen packages nothing used -
``fastapi`` and ``uvicorn`` among them. The statistics module hand-rolls exact
McNemar on ``math`` rather than take scipy, and BM25 is ``collections.Counter``.
A stdlib server is the house style, not a compromise.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from aegis.console.api import (
    ConsoleError,
    boot_payload,
    load_policy,
    run_scenario,
    run_turn,
)

__all__ = ["PAGE_PATH", "ConsoleServer", "build_server", "main", "serve"]

PAGE_PATH = Path(__file__).resolve().parent / "page.html"
"""The one asset served. A constant, never anything derived from a request."""

HOST = "127.0.0.1"
DEFAULT_PORT = 8017

MAX_BODY_BYTES = 256 * 1024
"""Enough for a pasted document, small enough that a stray upload cannot sit in
memory. The browser only ever posts a page and a handful of arguments."""


class ConsoleServer(ThreadingHTTPServer):
    """A server that remembers which policy the console is inspecting.

    The path lives here rather than in a module global so two servers can run
    in one process - which is exactly what the tests do - and so the handler
    reads it from the server it belongs to instead of from import state.
    """

    def __init__(self, address: tuple[str, int], policy_path: Path | None = None) -> None:
        super().__init__(address, ConsoleHandler)
        self.policy_path = policy_path

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Stay quiet when a browser simply hangs up.

        Closing a tab, navigating away, or a keep-alive connection timing out all
        surface as ``ConnectionResetError``/``ConnectionAbortedError`` inside the
        stdlib's own read loop, and the default handler prints a full traceback
        for each. On a demo whose terminal output is part of what the reader
        sees, a wall of tracebacks caused by ordinary browsing looks exactly like
        the console failing.

        Every other exception still gets the full treatment - a real bug must not
        be swallowed by a rule written for a closed socket.
        """
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


class ConsoleHandler(BaseHTTPRequestHandler):
    """Routes four paths and refuses everything else."""

    server_version = "AegisConsole/1.0"
    protocol_version = "HTTP/1.1"

    timeout = 30
    """Seconds a single request may hold its worker thread.

    Without it, a request declaring a ``Content-Length`` larger than the bytes it
    actually sends blocks forever in ``rfile.read`` - one idle thread per stalled
    connection, held indefinitely. Loopback-only and single-operator makes that a
    tidiness problem rather than a denial of service, but a server that never
    lets go of a socket is still a server that leaks.
    """

    @property
    def policy_path(self) -> Path | None:
        """The policy this console was started against, or None for the shipped one."""
        return cast(ConsoleServer, self.server).policy_path

    def do_GET(self) -> None:  # the stdlib names it, not us
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_page()
        elif path == "/api/boot":
            self._guarded(lambda: boot_payload(policy_path=self.policy_path))
        elif path.startswith("/api/scenario/"):
            key = path[len("/api/scenario/") :]
            self._guarded(lambda: run_scenario(key, policy_path=self.policy_path))
        else:
            self._send_json(404, {"error": f"no route for {path}"})

    def do_POST(self) -> None:  # the stdlib names it, not us
        path = self.path.split("?", 1)[0]
        if path != "/api/run":
            self._send_json(404, {"error": f"no route for {path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._refuse_without_reading(400, "Content-Length is not a number")
            return
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            # Otherwise read as length 0: the real payload was silently discarded
            # and the caller got a baffling "'tool' must be a non-empty string".
            self._refuse_without_reading(
                411, "chunked bodies are not supported; send a Content-Length"
            )
            return
        if length > MAX_BODY_BYTES:
            self._refuse_without_reading(413, f"body larger than {MAX_BODY_BYTES} bytes")
            return
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"body is not valid JSON: {exc}"})
            return
        except RecursionError:
            # Not a JSONDecodeError, so it escaped the clause above and killed the
            # response before _guarded could ever see it - the browser got a bare
            # network failure. A few thousand nested brackets is a malformed
            # request, not a bug in this console. api.MAX_DEPTH bounds the same
            # shape one layer in, for callers that never come through HTTP.
            self._refuse_without_reading(400, "body nests too deeply to parse")
            return
        self._guarded(lambda: run_turn(request, policy_path=self.policy_path))

    def _refuse_without_reading(self, status: int, message: str) -> None:
        """Answer an oversized or unparseable request, then hang up.

        The point of the size cap is that the body is never read - but this
        speaks HTTP/1.1, so leaving the connection open leaves those unread bytes
        in the socket, and the NEXT request on it gets parsed out of the middle
        of the old body. In a browser, which reuses connections, one oversized
        paste made every subsequent fetch fail with `414 URI Too Long` and a page
        of HTML garbage - a bug that looks like the console breaking at random,
        several actions after the cause.
        """
        self.close_connection = True
        self._send_json(status, {"error": message})

    # -- helpers -------------------------------------------------------------

    def _guarded(self, work: Callable[[], dict[str, Any]]) -> None:
        """Run a handler, separating a bad request from a bug in this console.

        A :class:`ConsoleError` is the caller's fault and comes back as a 400
        with the explanation.

        Anything else is ours, and gets BOTH halves of the treatment: the full
        traceback to stderr, where a developer will see it, and a 500 whose body
        names the exception, so the page can say what went wrong instead of
        showing a bare connection reset. Letting it propagate would abort the
        response mid-flight - on a keep-alive connection the browser reports a
        network failure, which is the least informative possible rendering of a
        bug in a page whose entire job is to display what happened.
        """
        try:
            self._send_json(200, work())
        except ConsoleError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # a demo server must not die on one bad request
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}", "bug": True})

    def _send_page(self) -> None:
        try:
            body = PAGE_PATH.read_bytes()
        except OSError:
            self._send_json(
                500,
                {"error": f"the console page is missing from the install at {PAGE_PATH}"},
            )
            return
        self._respond(200, "text/html; charset=utf-8", body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._respond(status, "application/json; charset=utf-8", body)

    def _respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page is entirely self-contained, so it needs no external origin at
        # all. Saying so costs one header and closes the gap between "we did not
        # add a CDN" and "a CDN cannot be added by accident".
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'; img-src data:; form-action 'none'; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """One tidy line per request instead of the stdlib's stderr noise.

        Two hazards, both reachable from a browser.

        The stdlib routes several different shapes through here - ``log_request``
        sends (requestline, code, size) while ``log_error`` sends (code, message)
        - so every argument is formatted rather than one position indexed.

        And ``log_error`` fires for a request line that failed to PARSE, at which
        point ``self.command`` and ``self.path`` have never been assigned.
        Reading them raised ``AttributeError`` from inside ``send_error``, so the
        400 was never written and the client received an empty response. Anyone
        who types ``https://`` against this port sends exactly that shape: the
        TLS ClientHello is not a request line.
        """
        command = getattr(self, "command", "-")
        path = getattr(self, "path", "-")
        sys.stderr.write(f"  {command} {path} {' '.join(str(a) for a in args)}\n")


def build_server(port: int = DEFAULT_PORT, *, policy_path: Path | None = None) -> ConsoleServer:
    """Bind the console on loopback. Separate from :func:`serve` so a test can
    bind port 0, talk to it, and shut it down without a signal handler."""
    return ConsoleServer((HOST, port), policy_path)


def serve(
    port: int = DEFAULT_PORT,
    *,
    open_browser: bool = True,
    policy_path: Path | None = None,
) -> int:
    httpd = build_server(port, policy_path=policy_path)
    url = f"http://{HOST}:{httpd.server_address[1]}/"
    print(f"Aegis console on {url}")
    print("  offline: no API key, no network, no model. Ctrl-C to stop.\n")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aegis-console",
        description="An offline console over the real Aegis defense layers.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="do not open a browser window")
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "inspect YOUR trust_tiers.yaml instead of the shipped one, so every "
            "verdict on the page is your deployment's policy rather than this "
            "repository's"
        ),
    )
    args = parser.parse_args(argv)
    if args.policy is not None:
        if not args.policy.is_file():
            parser.error(f"no policy file at {args.policy}")
        # Load it here rather than discover the problem on the page's first
        # request: a typo in an operator's YAML should stop the server with the
        # parse error on the terminal they are already looking at.
        try:
            load_policy(args.policy)
        except ConsoleError as exc:
            parser.error(str(exc))
    return serve(args.port, open_browser=not args.no_open, policy_path=args.policy)
