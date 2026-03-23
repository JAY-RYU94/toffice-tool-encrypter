"""
모델 응답에서 <tool_call> XML 블록을 파싱합니다.

코드/HTML 등 특수문자가 포함된 param 값도 안전하게 파싱하기 위해
여러 방식을 지원합니다:
  1. CDATA: <param name="x"><![CDATA[내용]]></param>
  2. 일반:  <param name="x">내용</param>  (내용에 XML 태그가 없는 경우)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# tool_call 블록 매칭 — 넉넉하게 잡고 내부에서 세부 파싱
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<tool\s+name=\"([^\"]+)\">(.*?)</tool>\s*</tool_call>",
    re.DOTALL,
)

# param 매칭 — CDATA 우선, 일반 폴백
_PARAM_CDATA_RE = re.compile(
    r"<param\s+name=\"([^\"]+)\">\s*<!\[CDATA\[(.*?)\]\]>\s*</param>",
    re.DOTALL,
)
_PARAM_PLAIN_RE = re.compile(
    r"<param\s+name=\"([^\"]+)\">(.*?)</param>",
    re.DOTALL,
)


@dataclass
class ParsedToolCall:
    name: str
    params: dict[str, str]


def parse_tool_calls(text: str) -> list[ParsedToolCall]:
    """응답 텍스트에서 모든 <tool_call> 블록을 파싱하여 반환합니다."""
    calls: list[ParsedToolCall] = []

    for block in _TOOL_CALL_RE.finditer(text):
        tool_name = block.group(1)
        body = block.group(2)
        params = _parse_params(body)
        calls.append(ParsedToolCall(name=tool_name, params=params))

    # 정규식으로 매칭 안 되는 경우 (코드에 </tool> 등이 포함된 경우)
    # 좀 더 관대한 파싱 시도
    if not calls and "<tool_call>" in text:
        calls = _parse_lenient(text)

    return calls


def _parse_params(body: str) -> dict[str, str]:
    """param 블록들을 파싱합니다. CDATA 방식 우선."""
    params: dict[str, str] = {}

    # 1) CDATA 방식 먼저
    for m in _PARAM_CDATA_RE.finditer(body):
        params[m.group(1)] = m.group(2)

    # 2) CDATA로 안 잡힌 param은 일반 방식으로
    for m in _PARAM_PLAIN_RE.finditer(body):
        name = m.group(1)
        if name not in params:  # CDATA로 이미 파싱된 건 건너뜀
            params[name] = _unescape_xml(m.group(2))

    return params


def _parse_lenient(text: str) -> list[ParsedToolCall]:
    """
    관대한 파싱 — 모델이 형식을 약간 벗어났을 때 복구를 시도합니다.
    <tool_call> 블록을 찾고 내부의 tool name과 param을 추출합니다.
    """
    calls: list[ParsedToolCall] = []

    # tool_call 시작/끝 위치를 직접 찾기
    start = 0
    while True:
        tc_start = text.find("<tool_call>", start)
        if tc_start == -1:
            break
        tc_end = text.find("</tool_call>", tc_start)
        if tc_end == -1:
            # 닫는 태그 없으면 텍스트 끝까지
            tc_end = len(text)
        block = text[tc_start:tc_end]

        # tool name 추출
        name_match = re.search(r'<tool\s+name="([^"]+)">', block)
        if name_match:
            tool_name = name_match.group(1)
            params = _parse_params(block)
            calls.append(ParsedToolCall(name=tool_name, params=params))

        start = tc_end + len("</tool_call>")

    return calls


def _unescape_xml(text: str) -> str:
    """기본적인 XML 이스케이프 문자를 복원합니다."""
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&apos;", "'")
    return text


def strip_tool_calls(text: str) -> str:
    """응답 텍스트에서 <tool_call> 블록을 제거하고 나머지 텍스트만 반환합니다."""
    # 정규식 매칭 + 수동 매칭 모두 처리
    result = _TOOL_CALL_RE.sub("", text)
    # 관대한 제거 — 정규식으로 안 잡힌 블록
    result = re.sub(r"<tool_call>.*?</tool_call>", "", result, flags=re.DOTALL)
    return result.strip()
