"""
모델 응답에서 <tool_call> XML 블록을 파싱합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedToolCall:
    name: str
    params: dict[str, str]


def parse_tool_calls(text: str) -> list[ParsedToolCall]:
    """응답 텍스트에서 모든 <tool_call> 블록을 파싱하여 반환합니다."""
    calls: list[ParsedToolCall] = []

    for block in re.finditer(
        r"<tool_call>\s*<tool\s+name=\"([^\"]+)\">(.*?)</tool>\s*</tool_call>",
        text,
        re.DOTALL,
    ):
        tool_name = block.group(1)
        body = block.group(2)
        params: dict[str, str] = {}
        for param in re.finditer(
            r"<param\s+name=\"([^\"]+)\">(.*?)</param>",
            body,
            re.DOTALL,
        ):
            params[param.group(1)] = param.group(2)
        calls.append(ParsedToolCall(name=tool_name, params=params))

    return calls


def strip_tool_calls(text: str) -> str:
    """응답 텍스트에서 <tool_call> 블록을 제거하고 나머지 텍스트만 반환합니다."""
    return re.sub(
        r"<tool_call>.*?</tool_call>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
