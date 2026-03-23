"""
Tool 실행기.

파싱된 tool call을 실제로 실행하고 결과를 반환합니다.
"""

from __future__ import annotations

import glob as glob_module
import os
import re
import subprocess
from dataclasses import dataclass


@dataclass
class ToolCall:
    """파싱된 tool call."""
    name: str
    params: dict[str, str]


@dataclass
class ToolResult:
    """Tool 실행 결과."""
    tool_name: str
    success: bool
    output: str


class ToolExecutor:
    """등록된 도구들을 실행합니다."""

    def __init__(self, working_dir: str | None = None):
        self.working_dir = working_dir or os.getcwd()

    def execute(self, call: ToolCall) -> ToolResult:
        handler = getattr(self, f"_tool_{call.name}", None)
        if handler is None:
            return ToolResult(call.name, False, f"Unknown tool: {call.name}")
        try:
            output = handler(call.params)
            return ToolResult(call.name, True, output)
        except Exception as e:
            return ToolResult(call.name, False, f"Error: {e}")

    # ─── Tool 구현 ───

    # 위험한 패턴 — 실행 전 경고를 포함하여 반환
    _DANGEROUS_PATTERNS = [
        (r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|)(/|~|\$HOME)", "rm on system/home directory"),
        (r"\bmkfs\b", "filesystem format"),
        (r"\bdd\s+.*of\s*=\s*/dev/", "raw device write"),
        (r">\s*/dev/sd[a-z]", "raw device overwrite"),
        (r"\bcurl\b.*\|\s*(ba)?sh", "pipe remote script to shell"),
        (r"\bwget\b.*\|\s*(ba)?sh", "pipe remote script to shell"),
        (r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;", "fork bomb"),
    ]

    def _tool_bash(self, params: dict[str, str]) -> str:
        command = params.get("command", "")
        if not command:
            return "Error: command is required"

        # 위험 명령 패턴 검사
        for pattern, reason in self._DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return (
                    f"Error: potentially dangerous command blocked ({reason}).\n"
                    f"Command: {command}\n"
                    f"If you need to run this, ask the user for confirmation first via ask_user."
                )

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.working_dir,
            timeout=120,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"

    def _tool_read_file(self, params: dict[str, str]) -> str:
        path = self._resolve_path(params.get("path", ""))
        if not os.path.isfile(path):
            return f"Error: file not found: {path}"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        offset = int(params.get("offset", "0"))
        limit = int(params.get("limit", "2000"))
        selected = lines[offset : offset + limit]

        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i:>6}\t{line.rstrip()}")
        return "\n".join(numbered)

    def _tool_write_file(self, params: dict[str, str]) -> str:
        path = self._resolve_path(params.get("path", ""))
        content = params.get("content", "")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written: {path}"

    def _tool_edit_file(self, params: dict[str, str]) -> str:
        path = self._resolve_path(params.get("path", ""))
        if not os.path.isfile(path):
            return f"Error: file not found: {path}"
        old_string = params.get("old_string", "")
        new_string = params.get("new_string", "")
        if not old_string:
            return "Error: old_string is required"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1:
            return f"Error: old_string found {count} times. Provide more context to make it unique."

        content = content.replace(old_string, new_string, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File edited: {path}"

    def _tool_grep(self, params: dict[str, str]) -> str:
        pattern = params.get("pattern", "")
        path = self._resolve_path(params.get("path", "."))
        include = params.get("include", "")

        cmd = ["grep", "-rn", pattern, path]
        if include:
            cmd.insert(2, f"--include={include}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.working_dir,
            timeout=30,
        )
        output = result.stdout.strip()
        if not output:
            return "No matches found."
        # 결과가 너무 길면 잘라냄
        lines = output.split("\n")
        if len(lines) > 100:
            return "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
        return output

    def _tool_glob(self, params: dict[str, str]) -> str:
        pattern = params.get("pattern", "")
        path = self._resolve_path(params.get("path", "."))
        full_pattern = os.path.join(path, pattern)
        matches = sorted(glob_module.glob(full_pattern, recursive=True))
        if not matches:
            return "No files matched."
        # 상대 경로로 표시
        rel = [os.path.relpath(m, self.working_dir) for m in matches]
        if len(rel) > 200:
            return "\n".join(rel[:200]) + f"\n... ({len(rel) - 200} more files)"
        return "\n".join(rel)

    def _resolve_path(self, path: str) -> str:
        if not path:
            return self.working_dir
        if os.path.isabs(path):
            return path
        return os.path.join(self.working_dir, path)
