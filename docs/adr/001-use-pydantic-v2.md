# ADR-001: 使用 Pydantic V2 作为数据契约层

## 决策
采用 Pydantic V2 定义所有模块间的数据接口，禁止裸 dict 跨模块传递

## 理由
- LLM 应用数据流复杂，弱类型 dict 极易导致 `KeyError` 和类型错误。
- Pydantic V2 性能卓越（Rust 核心），适合处理海量分块。
- 强制运行时校验，是构建健壮系统的基石。