"""
OpenAI 호환 게이트웨이 API 클라이언트.

사내 게이트웨이가 OpenAI 호환 API를 제공하지만 tools 파라미터가 차단된 환경에서,
tools 없이 순수 chat completion만 사용합니다.
Tool 정의는 system prompt에 포함되어 모델이 XML로 tool call을 출력합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Generator

from openai import OpenAI


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


class GatewayClient:
    """OpenAI 호환 게이트웨이 클라이언트. tools 파라미터 없이 동작."""

    def __init__(self, config: GatewayConfig | None = None):
        self.config = config or GatewayConfig.from_env()
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
    ) -> str:
        """
        게이트웨이로 메시지를 보내고 응답 텍스트를 반환합니다.
        tools 파라미터 없이 순수 텍스트 요청만 사용합니다.
        """
        full_messages = self._build_messages(messages, system)

        # tools 파라미터는 의도적으로 포함하지 않음
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
    ) -> Generator[str, None, None]:
        """스트리밍 응답. 각 청크의 텍스트를 yield합니다."""
        full_messages = self._build_messages(messages, system)

        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

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
