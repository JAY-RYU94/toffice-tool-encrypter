"""
컨텍스트 윈도우 관리.

대화 기록이 너무 길어지면 오래된 tool 결과를 요약하여 토큰을 절약합니다.
"""

from __future__ import annotations

from typing import Any


# 대략적인 토큰 추정 (1 토큰 ≈ 4 chars, 보수적으로)
def _estimate_tokens(text: str) -> int:
    return len(text) // 3


def _message_tokens(msg: dict[str, Any]) -> int:
    content = msg.get("content", "")
    return _estimate_tokens(content) + 4  # role overhead


def total_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_message_tokens(m) for m in messages)


def trim_messages(
    messages: list[dict[str, Any]],
    max_tokens: int = 80_000,
    keep_recent: int = 10,
) -> list[dict[str, Any]]:
    """
    메시지 리스트가 max_tokens를 초과하면 오래된 메시지를 정리합니다.

    전략:
    1. 최근 keep_recent개의 메시지는 항상 유지
    2. 오래된 tool_result 메시지의 출력을 요약(잘라냄)
    3. 그래도 초과하면 가장 오래된 메시지부터 제거
    """
    current = total_tokens(messages)
    if current <= max_tokens:
        return messages

    # Phase 1: 오래된 tool_result 내용을 축약
    cutoff = len(messages) - keep_recent
    for i in range(cutoff):
        msg = messages[i]
        content = msg.get("content", "")

        # tool_result가 포함된 긴 메시지 축약
        if "<tool_result" in content and _message_tokens(msg) > 500:
            # 결과를 짧게 요약
            lines = content.split("\n")
            if len(lines) > 20:
                truncated = "\n".join(lines[:10]) + "\n... (truncated for context)\n" + "\n".join(lines[-5:])
                messages[i] = {**msg, "content": truncated}

    current = total_tokens(messages)
    if current <= max_tokens:
        return messages

    # Phase 2: 그래도 초과하면 오래된 메시지 제거하되 첫 user 메시지는 유지
    while len(messages) > keep_recent and total_tokens(messages) > max_tokens:
        # 인덱스 1부터 제거 (0번은 첫 user 메시지일 수 있으므로)
        if len(messages) > keep_recent + 1:
            messages.pop(1)
        else:
            break

    return messages
