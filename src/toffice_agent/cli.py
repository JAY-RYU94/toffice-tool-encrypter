"""
CLI 엔트리포인트.

사용법:
    GATEWAY_BASE_URL=https://your-gateway/v1 \
    GATEWAY_API_KEY=your-key \
    GATEWAY_MODEL=gpt-4o \
    python -m toffice_agent
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .agent import Agent
from .gateway_client import GatewayConfig
from .parser import ParsedToolCall
from .tools.executor import ToolResult

console = Console()


def _on_tool_call(tc: ParsedToolCall) -> None:
    """Tool 실행 전 표시."""
    params_display = []
    for k, v in tc.params.items():
        display_val = repr(v[:120] + "...") if len(v) > 120 else repr(v)
        params_display.append(f"  {k}={display_val}")
    params_str = "\n".join(params_display)
    console.print(
        Panel(
            f"[bold yellow]{tc.name}[/]\n{params_str}",
            title="Tool Call",
            border_style="yellow",
        )
    )


def _on_tool_result(result: ToolResult) -> None:
    """Tool 결과 출력."""
    style = "green" if result.success else "red"
    output = result.output
    if len(output) > 1000:
        output = output[:1000] + f"\n... ({len(result.output) - 1000} chars truncated)"
    console.print(
        Panel(output, title=f"{result.tool_name}", border_style=style)
    )


def _on_text(text: str) -> None:
    """모델의 텍스트 응답 출력."""
    console.print()
    console.print(Markdown(text))


def _on_ask_user(question: str) -> str:
    """ask_user 도구 콜백 - 사용자에게 질문을 보여주고 응답을 받음."""
    console.print()
    console.print(
        Panel(
            Markdown(question),
            title="Agent asks",
            border_style="cyan",
        )
    )
    try:
        reply = console.input("[bold cyan]Your reply > [/]").strip()
    except (EOFError, KeyboardInterrupt):
        reply = "cancel"
    return reply


def main():
    config = GatewayConfig.from_env()
    console.print(
        Panel(
            f"Gateway: [cyan]{config.base_url}[/]\n"
            f"Model:   [cyan]{config.model}[/]\n\n"
            f"Commands: [dim]exit, /reset[/]",
            title="toffice-agent",
            border_style="blue",
        )
    )

    agent = Agent(
        config=config,
        on_tool_call=_on_tool_call,
        on_tool_result=_on_tool_result,
        on_text=_on_text,
        on_ask_user=_on_ask_user,
    )

    while True:
        try:
            console.print()
            user_input = console.input("[bold green]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            console.print("Bye!")
            break
        if user_input.lower() == "/reset":
            agent.reset()
            console.print("[dim]Conversation reset.[/]")
            continue

        agent.run(user_input)


if __name__ == "__main__":
    main()
