# src/llm_doc_parser/parsers/base.py
from __future__ import annotations

import abc
from pathlib import Path

from llm_doc_parser.models import ParseResult


class BaseParser(abc.ABC):
    """
    文档解析器的抽象基类。
    所有具体解析器（Docling, MinerU, Unstructured）必须实现此接口。

    设计意图：
    - 业务层只依赖此抽象，不感知底层库
    - 方便后续做 A/B 测试和多解析器对比评测
    """

    @abc.abstractmethod
    async def parse(self, file_path: Path) -> ParseResult:
        """
        异步解析文档，返回强类型 ParseResult。

        Args:
            file_path: 文档文件路径

        Returns:
            ParseResult: 包含分块列表、元数据和性能指标的统一结果

        Raises:
            FileNotFoundError: 文件不存在
            ParseError: 解析失败（非致命错误应记录在 warnings 中）
        """
        ...
