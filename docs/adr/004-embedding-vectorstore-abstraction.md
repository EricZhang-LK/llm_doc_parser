# ADR-004: Embedding 与 VectorStore 双重抽象

## 状态
已采纳 (2026-06-04)

## 决策
1. 定义 BaseEmbedder 和 BaseVectorStore 两个独立抽象基类
2. Embedder 仅暴露批量接口，强制批处理思维
3. VectorStore 的 upsert 分离 chunks 与 vectors，保持领域模型纯净
4. D4 仅提供 InMemory 实现用于契约验证，生产实现延后

## 理由
- Embedding 模型和向量数据库是 RAG 中最易变的组件
- 双重解耦支持独立演进和 A/B 测试
- 契约测试确保新实现的正确性无需额外验证成本

## 后果
- (+) 供应商无关，迁移成本低，测试覆盖率高
- (-) 抽象层增加少量间接性，调试时需多跳一层
- (!) filters 参数暂用 dict，后续可按需引入结构化 Filter DSL