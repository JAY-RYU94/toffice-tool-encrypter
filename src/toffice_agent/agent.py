"""
에이전트 루프.

Claude Code 스타일 워크플로우:
1. 사용자 요청을 받는다
2. 모델이 코드베이스를 탐색하고 계획을 수립한다
3. 계획을 ask_user 도구로 사용자에게 제시하고 승인을 받는다
4. 승인 후 계획에 따라 코드를 작성/수정한다
5. 완료 시 최종 응답을 출력한다
"""

from __future__ import annotations

from typing import Any, Callable

from .gateway_client import GatewayClient, GatewayConfig
from .parser import ParsedToolCall, parse_tool_calls, strip_tool_calls
from .tools.definitions import BUILTIN_TOOLS, tools_to_system_prompt
from .tools.executor import ToolCall, ToolExecutor, ToolResult

SYSTEM_PROMPT_TEMPLATE = """\
You are a highly capable software engineering assistant, similar to Claude Code.
You help users with coding tasks by reading, writing, and editing files,
running commands, and searching codebases.

## Workflow

When the user gives you a task, follow this workflow:

1. **Explore**: First, use tools (read_file, grep, glob, bash) to understand the codebase and gather context.
2. **Plan**: Create a detailed step-by-step plan for the task. Present the plan to the user using the `ask_user` tool and wait for approval.
3. **Execute**: After the user approves, execute the plan step by step using tools.
4. **Verify**: After making changes, verify your work (run tests, check for errors).
5. **Report**: Provide a concise summary of what was done.

## Important Rules

- ALWAYS explore the codebase first before making changes.
- ALWAYS present a plan and get user approval before writing/editing code.
- Use `ask_user` to ask clarifying questions or present plans.
- Be concise and direct in your responses.
- When you're done with the task, respond with plain text (no tool_call blocks).
- If the user says "no" or rejects your plan, revise it based on their feedback.

{tool_spec}
"""


class Agent:
    """Prompt 기반 tool use 에이전트 (Claude Code 스타일)."""

    def __init__(
        self,
        config: GatewayConfig | None = None,
        working_dir: str | None = None,
        max_iterations: int = 50,
        on_tool_call: Callable[[ParsedToolCall], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_ask_user: Callable[[str], str] | None = None,
    ):
        self.client = GatewayClient(config)
        self.executor = ToolExecutor(working_dir)
        self.max_iterations = max_iterations
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_spec=tools_to_system_prompt(BUILTIN_TOOLS)
        )
        self.messages: list[dict[str, Any]] = []

        # 콜백
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_text = on_text
        self.on_ask_user = on_ask_user  # ask_user 도구 호출 시 유저 입력을 받는 콜백

    def run(self, user_message: str) -> str:
        """사용자 메시지를 받아 에이전트 루프를 실행하고 최종 응답을 반환합니다."""
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(self.max_iterations):
            response = self.client.chat(self.messages, system=self.system_prompt)
            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                # 최종 응답
                self.messages.append({"role": "assistant", "content": response})
                if self.on_text:
                    self.on_text(response)
                return response

            # tool call이 있는 경우
            self.messages.append({"role": "assistant", "content": response})

            # 텍스트 부분이 있으면 출력
            plain_text = strip_tool_calls(response)
            if plain_text and self.on_text:
                self.on_text(plain_text)

            # 각 tool call 실행
            results: list[str] = []
            for tc in tool_calls:
                # ask_user는 특별 처리
                if tc.name == "ask_user":
                    result_text = self._handle_ask_user(tc)
                    results.append(result_text)
                    continue

                if self.on_tool_call:
                    self.on_tool_call(tc)

                result = self.executor.execute(
                    ToolCall(name=tc.name, params=tc.params)
                )
                if self.on_tool_result:
                    self.on_tool_result(result)

                status = "success" if result.success else "error"
                results.append(
                    f"<tool_result name=\"{tc.name}\" status=\"{status}\">\n"
                    f"{result.output}\n"
                    f"</tool_result>"
                )

            # tool 결과를 user 메시지로 추가
            self.messages.append({
                "role": "user",
                "content": "\n\n".join(results),
            })

        return "Error: max iterations reached."

    def _handle_ask_user(self, tc: ParsedToolCall) -> str:
        """ask_user 도구를 처리합니다."""
        question = tc.params.get("question", "")
        if self.on_ask_user:
            user_reply = self.on_ask_user(question)
        else:
            user_reply = input(f"\n{question}\n> ")
        return (
            f"<tool_result name=\"ask_user\" status=\"success\">\n"
            f"User response: {user_reply}\n"
            f"</tool_result>"
        )

    def reset(self):
        """대화 기록을 초기화합니다."""
        self.messages.clear()
