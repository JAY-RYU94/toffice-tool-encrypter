"""
OpenAI 호환 게이트웨이 API 클라이언트.

사내 게이트웨이가 OpenAI 호환 API를 제공하지만 tools 파라미터가 차단된 환경에서,
tools 없이 순수 chat completion만 사용합니다.
Tool 정의는 system prompt에 포함되어 모델이 XML로 tool call을 출력합니다.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Generator

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class GatewayConfig:
    """게이트웨이 API 설정."""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = 8192

    @classmethod
    def from_env(cls) -> GatewayConfig:
        return cls(
            base_url=os.environ.get("GATEWAY_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("GATEWAY_API_KEY", "no-key"),
            model=os.environ.get("GATEWAY_MODEL", "gpt-4o"),
            max_tokens=int(os.environ.get("GATEWAY_MAX_TOKENS", "8192")),
        )


# 재시도 가능한 에러 타입
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 4, 8]


class GatewayClient:
    """OpenAI 호환 게이트웨이 클라이언트. tools 파라미터 없이 동작."""

    def __init__(self, config: GatewayConfig | None = None):
        self.config = config or GatewayConfig.from_env()
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=120.0,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
    ) -> str:
        """동기 응답. 전체 텍스트를 한번에 반환합니다."""
        full_messages = self._build_messages(messages, system)
        resp = self._call_with_retry(full_messages, stream=False)
        return resp.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
    ) -> Generator[str, None, None]:
        """스트리밍 응답. 각 청크의 텍스트를 yield합니다."""
        full_messages = self._build_messages(messages, system)
        stream = self._call_with_retry(full_messages, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    def _call_with_retry(self, messages: list[dict[str, Any]], stream: bool):
        """재시도 로직 포함 API 호출."""
        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    stream=stream,
                )
            except _RETRYABLE as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(f"API error (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
            except KeyboardInterrupt:
                raise
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _build_messages(
        messages: list[dict[str, Any]],
        system: str,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if system:
            result.append({"role": "system", "content": system})
        result.extend(messages)
        return result

    def close(self):
        self._client.close()
