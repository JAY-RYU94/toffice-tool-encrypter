"""
Tool 정의 및 system prompt 생성.

각 tool의 스펙을 정의하고, 이를 system prompt에 삽입할 XML로 변환합니다.
게이트웨이의 tools 파라미터 대신 이 방식을 사용합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)


# ─── 기본 제공 도구 목록 ───

BUILTIN_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="bash",
        description="셸 명령어를 실행하고 결과를 반환합니다.",
        parameters=[
            ToolParameter("command", "string", "실행할 셸 명령어"),
        ],
    ),
    ToolDefinition(
        name="read_file",
        description="파일의 내용을 읽어 반환합니다.",
        parameters=[
            ToolParameter("path", "string", "읽을 파일의 절대 경로 또는 상대 경로"),
            ToolParameter("offset", "integer", "읽기 시작할 줄 번호 (0부터)", required=False),
            ToolParameter("limit", "integer", "읽을 최대 줄 수", required=False),
        ],
    ),
    ToolDefinition(
        name="write_file",
        description="파일에 내용을 작성합니다. 기존 파일은 덮어씁니다.",
        parameters=[
            ToolParameter("path", "string", "작성할 파일 경로"),
            ToolParameter("content", "string", "파일에 작성할 내용"),
        ],
    ),
    ToolDefinition(
        name="edit_file",
        description="파일에서 특정 문자열을 찾아 다른 문자열로 교체합니다.",
        parameters=[
            ToolParameter("path", "string", "편집할 파일 경로"),
            ToolParameter("old_string", "string", "찾을 문자열 (정확히 일치해야 함)"),
            ToolParameter("new_string", "string", "교체할 문자열"),
        ],
    ),
    ToolDefinition(
        name="grep",
        description="파일들에서 정규식 패턴을 검색합니다.",
        parameters=[
            ToolParameter("pattern", "string", "검색할 정규식 패턴"),
            ToolParameter("path", "string", "검색할 디렉토리 또는 파일 경로", required=False),
            ToolParameter("include", "string", "포함할 파일 패턴 (예: *.py)", required=False),
        ],
    ),
    ToolDefinition(
        name="glob",
        description="글로브 패턴으로 파일을 검색합니다.",
        parameters=[
            ToolParameter("pattern", "string", "검색할 글로브 패턴 (예: **/*.py)"),
            ToolParameter("path", "string", "검색 시작 디렉토리", required=False),
        ],
    ),
    ToolDefinition(
        name="ask_user",
        description="사용자에게 질문하거나 확인을 요청합니다. 계획을 제시하고 승인을 받을 때 사용합니다.",
        parameters=[
            ToolParameter("question", "string", "사용자에게 보낼 질문 또는 계획"),
        ],
    ),
]


def tools_to_system_prompt(tools: list[ToolDefinition] | None = None) -> str:
    """Tool 정의를 system prompt에 삽입할 XML 문자열로 변환합니다."""
    if tools is None:
        tools = BUILTIN_TOOLS

    lines: list[str] = []
    lines.append("You have access to the following tools. To use a tool, respond with an XML block in exactly this format:")
    lines.append("")
    lines.append("<tool_call>")
    lines.append('<tool name="TOOL_NAME">')
    lines.append('<param name="PARAM_NAME">VALUE</param>')
    lines.append("</tool>")
    lines.append("</tool_call>")
    lines.append("")
    lines.append("You may call multiple tools in a single response by including multiple <tool_call> blocks.")
    lines.append("After each tool execution, you will receive the result in a <tool_result> block.")
    lines.append("When you have completed the task and no more tool calls are needed, respond with your final answer WITHOUT any <tool_call> blocks.")
    lines.append("")
    lines.append("Available tools:")
    lines.append("")

    for tool in tools:
        lines.append(f"### {tool.name}")
        lines.append(f"Description: {tool.description}")
        if tool.parameters:
            lines.append("Parameters:")
            for p in tool.parameters:
                req = " (required)" if p.required else " (optional)"
                lines.append(f"  - {p.name} ({p.type}){req}: {p.description}")
        lines.append("")

    return "\n".join(lines)
