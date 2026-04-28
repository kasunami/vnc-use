#!/usr/bin/env python3
"""Health check for a local vnc-use Docker Compose deployment."""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
from dataclasses import asdict, dataclass

from fastmcp import Client


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _tcp_check(name: str, host: str, port: int, timeout_s: float) -> CheckResult:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return CheckResult(name=name, ok=True, detail=f"{host}:{port} reachable")
    except OSError as exc:
        return CheckResult(name=name, ok=False, detail=f"{host}:{port} unreachable: {exc}")


async def _mcp_tools_check(url: str) -> CheckResult:
    try:
        async with Client(url) as client:
            tools = await client.list_tools()
        names = sorted(tool.name for tool in tools)
        if "execute_vnc_task" not in names:
            return CheckResult("mcp_tools", False, f"execute_vnc_task missing; tools={names}")
        return CheckResult("mcp_tools", True, f"tools={names}")
    except Exception as exc:
        return CheckResult("mcp_tools", False, str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8001/mcp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=8001)
    parser.add_argument("--novnc-port", type=int, default=6901)
    parser.add_argument("--vnc-port", type=int, default=5901)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    results = [
        _tcp_check("mcp_tcp", args.host, args.mcp_port, args.timeout_s),
        _tcp_check("novnc_tcp", args.host, args.novnc_port, args.timeout_s),
        _tcp_check("vnc_tcp", args.host, args.vnc_port, args.timeout_s),
        asyncio.run(_mcp_tools_check(args.mcp_url)),
    ]
    ok = all(result.ok for result in results)
    print(json.dumps({"ok": ok, "checks": [asdict(result) for result in results]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
