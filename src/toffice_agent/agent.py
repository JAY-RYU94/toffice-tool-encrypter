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

from .context import total_tokens, trim_messages
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


class AgentInterrupted(Exception):
    """사용자가 Ctrl+C로 현재 작업을 중단했을 때."""
    pass


class Agent:
    """Prompt 기반 tool use 에이전트 (Claude Code 스타일)."""

    def __init__(
        self,
        config: GatewayConfig | None = None,
        working_dir: str | None = None,
        max_iterations: int = 50,
        max_context_tokens: int = 80_000,
        use_stream: bool = True,
        on_tool_call: Callable[[ParsedToolCall], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_stream_chunk: Callable[[str], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_ask_user: Callable[[str], str] | None = None,
    ):
        self.client = GatewayClient(config)
        self.executor = ToolExecutor(working_dir)
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.use_stream = use_stream
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_spec=tools_to_system_prompt(BUILTIN_TOOLS)
        )
        self.messages: list[dict[str, Any]] = []

        # 콜백
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_stream_chunk = on_stream_chunk  # 스트리밍 중 각 청크
        self.on_text = on_text                  # 최종 텍스트 (스트리밍 미사용 시)
        self.on_ask_user = on_ask_user

    def run(self, user_message: str) -> str:
        """사용자 메시지를 받아 에이전트 루프를 실행하고 최종 응답을 반환합니다."""
        self.messages.append({"role": "user", "content": user_message})

        for iteration in range(self.max_iterations):
            # 컨텍스트 윈도우 관리
            self.messages = trim_messages(
                self.messages,
                max_tokens=self.max_context_tokens,
            )

            try:
                response = self._get_response()
            except KeyboardInterrupt:
                raise AgentInterrupted()
            except Exception as e:
                error_msg = f"API error: {e}"
                if self.on_text:
                    self.on_text(error_msg)
                return error_msg

            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                # 최종 응답
                self.messages.append({"role": "assistant", "content": response})
                # 스트리밍이 아닌 경우에만 on_text 호출 (스트리밍은 이미 출력됨)
                if not self.use_stream and self.on_text:
                    self.on_text(response)
                return response

            # tool call이 있는 경우
            self.messages.append({"role": "assistant", "content": response})

            # 텍스트 부분이 있으면 출력 (스트리밍이 아닌 경우)
            if not self.use_stream:
                plain_text = strip_tool_calls(response)
                if plain_text and self.on_text:
                    self.on_text(plain_text)

            # 각 tool call 실행
            tool_results: list[str] = []
            user_replies: list[str] = []
            try:
                for tc in tool_calls:
                    if tc.name == "ask_user":
                        reply_text = self._handle_ask_user(tc)
                        user_replies.append(reply_text)
                    else:
                        result_text = self._execute_tool(tc)
                        tool_results.append(result_text)
            except KeyboardInterrupt:
                tool_results.append("[Interrupted by user]")
                raise AgentInterrupted()

            # tool 결과는 시스템 메시지로 추가 (사용자 발화가 아님을 명시)
            if tool_results:
                tool_output = "\n\n".join(tool_results)
                self.messages.append({
                    "role": "user",
                    "content": f"[Tool execution results - not from the user]\n\n{tool_output}",
                })

            # ask_user 응답은 실제 사용자 발화이므로 별도 user 메시지로
            if user_replies:
                self.messages.append({
                    "role": "user",
                    "content": "\n\n".join(user_replies),
                })

        return "Error: max iterations reached."

    def _get_response(self) -> str:
        """스트리밍 여부에 따라 응답을 받습니다."""
        if self.use_stream:
            chunks: list[str] = []
            for chunk in self.client.chat_stream(self.messages, system=self.system_prompt):
                chunks.append(chunk)
                if self.on_stream_chunk:
                    self.on_stream_chunk(chunk)
            return "".join(chunks)
        else:
            return self.client.chat(self.messages, system=self.system_prompt)

    def _execute_tool(self, tc: ParsedToolCall) -> str:
        """단일 tool call을 실행하고 결과 XML을 반환합니다."""
        if self.on_tool_call:
            self.on_tool_call(tc)

        result = self.executor.execute(
            ToolCall(name=tc.name, params=tc.params)
        )
        if self.on_tool_result:
            self.on_tool_result(result)

        status = "success" if result.success else "error"
        return (
            f"<tool_result name=\"{tc.name}\" status=\"{status}\">\n"
            f"{result.output}\n"
            f"</tool_result>"
        )

    def _handle_ask_user(self, tc: ParsedToolCall) -> str:
        """ask_user 도구를 처리합니다. 사용자의 실제 응답을 반환합니다."""
        question = tc.params.get("question", "")
        if self.on_ask_user:
            user_reply = self.on_ask_user(question)
        else:
            user_reply = input(f"\n{question}\n> ")
        return user_reply

    def reset(self):
        """대화 기록을 초기화합니다."""
        self.messages.clear()
