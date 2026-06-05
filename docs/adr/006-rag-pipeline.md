# ADR-006: 采用 Jinja2 作为 Prompt 模板引擎与 RAG Pipeline 架构

## 状态
已采纳 (2026-06-05)

## 决策
1. 使用 Jinja2 管理 Prompt 模板，模板文件存放在 `prompts/` 目录。
2. RAGPipeline 采用依赖注入模式，解耦 Embedder、VectorStore 和 LLM。
3. 检索失败或为空时，Pipeline 直接返回兜底文案，不调用 LLM 以节省成本。

## 理由
- Jinja2 生态成熟，支持复杂的逻辑渲染，且性能极高。
- 依赖注入使得 RAGPipeline 极易测试（可轻松 Mock 各个组件）。
- 明确的异常处理流程提升了系统的鲁棒性。

## 后果
- (+) Prompt 修改无需重新部署代码。
- (+) 核心业务流程清晰，易于扩展（如增加重排序 Rerank 步骤）。
- (-) 增加了 Jinja2 依赖。