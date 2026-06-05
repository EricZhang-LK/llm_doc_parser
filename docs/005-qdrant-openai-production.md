# ADR-005: 选择 Qdrant + OpenAI Embedding 作为默认生产栈

## 状态
已采纳 (2026-06-05)

## 决策
1. Qdrant 作为首个生产向量存储，支持本地内存模式用于测试
2. OpenAI text-embedding-3-small 作为默认 Embedding，支持动态维度
3. 适配器内置重试、批处理分片、确定性 ID、payload 命名空间隔离

## 理由
- Qdrant: Rust 高性能、Payload Filter 原生支持、开发者体验最佳
- OpenAI E3S: 性价比最优，MRL 支持降维，生态成熟
- 生产级特性（重试/幂等/隔离）在适配器层统一解决，业务层保持简洁

## 后果
- (+) 开箱即用的生产级 RAG 存储层，CI 友好
- (-) 依赖 OpenAI API（可通过替换 Embedder 缓解）
- (!) Qdrant 本地内存模式仅用于测试，生产环境必须用持久化部署