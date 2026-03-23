"""
CLI 엔트리포인트.

사용법:
    GATEWAY_BASE_URL=https://your-gateway/v1 \
    GATEWAY_API_KEY=your-key \
    GATEWAY_MODEL=gpt-4o \
    python -m toffice_agent
"""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from .agent import Agent, AgentInterrupted
from .gateway_client import GatewayConfig
from .parser import ParsedToolCall
from .tools.executor import ToolResult

console = Console()

# 스트리밍 출력용 버퍼
_stream_buf: list[str] = []
_live: Live | None = None


def _on_stream_chunk(chunk: str) -> None:
    """스트리밍 청크를 실시간으로 표시합니다."""
    global _live
    _stream_buf.append(chunk)
    text = "".join(_stream_buf)

    if _live is None:
        _live = Live(Markdown(text), console=console, refresh_per_second=8)
        _live.start()
    else:
        _live.update(Markdown(text))


def _flush_stream() -> None:
    """스트리밍 버퍼를 정리합니다."""
    global _live
    if _live is not None:
        _live.stop()
        _live = None
    _stream_buf.clear()


def _on_tool_call(tc: ParsedToolCall) -> None:
    """Tool 실행 전 표시."""
    _flush_stream()
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
    """모델의 텍스트 응답 출력 (비스트리밍 모드용)."""
    console.print()
    console.print(Markdown(text))


def _on_ask_user(question: str) -> str:
    """ask_user 도구 콜백 — 사용자에게 질문을 보여주고 응답을 받음."""
    _flush_stream()
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
            f"Commands: [dim]exit, /reset, /help[/]",
            title="toffice-agent",
            border_style="blue",
        )
    )

    agent = Agent(
        config=config,
        use_stream=True,
        on_tool_call=_on_tool_call,
        on_tool_result=_on_tool_result,
        on_stream_chunk=_on_stream_chunk,
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

        cmd = user_input.lower()
        if cmd in ("exit", "quit"):
            console.print("Bye!")
            break
        if cmd == "/reset":
            agent.reset()
            console.print("[dim]Conversation reset.[/]")
            continue
        if cmd == "/help":
            console.print(
                Panel(
                    "[bold]Commands:[/]\n"
                    "  /reset  - Clear conversation history\n"
                    "  /help   - Show this help\n"
                    "  exit    - Quit\n\n"
                    "[bold]Environment variables:[/]\n"
                    "  GATEWAY_BASE_URL  - Gateway API endpoint\n"
                    "  GATEWAY_API_KEY   - API key\n"
                    "  GATEWAY_MODEL     - Model name\n"
                    "  GATEWAY_MAX_TOKENS - Max response tokens",
                    title="Help",
                    border_style="blue",
                )
            )
            continue

        try:
            agent.run(user_input)
        except AgentInterrupted:
            console.print("\n[yellow]Interrupted. You can continue or type a new request.[/]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/]")
        finally:
            _flush_stream()


if __name__ == "__main__":
    main()
