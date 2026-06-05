# src/llm_doc_parser/prompt_engine.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


class PromptEngine:
    """
    基于 Jinja2 的 Prompt 模板引擎。
    支持从文件系统加载模板，并提供上下文渲染功能。
    """

    def __init__(self, template_dir: str = "prompts") -> None:
        self._env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / template_dir),
            trim_blocks=True,  # 自动去除块后的第一个换行符
            lstrip_blocks=True,  # 自动去除块前的空白
        )

    def render(self, template_name: str, **context: Any) -> str:
        """
        渲染指定的 Prompt 模板。
        
        Args:
            template_name: 模板文件名（如 "rag_qa.jinja2"）
            **context: 渲染所需的上下文变量（如 question, context）
        """
        template = self._env.get_template(template_name)
        return template.render(**context)