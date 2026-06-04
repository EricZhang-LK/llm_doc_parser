# ADR-003: 采用语义感知分块 + tiktoken 精确计量

## 状态
已采纳 (2026-06-04)

## 背景
D2 的 Docling 解析器产出了结构化文本，但固定长度分块会导致语义断裂，
且 len//4 的 token 估算在中文/代码场景下误差达 30-50%，影响成本和上下文管理。

## 决策
1. 采用"结构优先 → 句子边界 → 长度兜底"的三层分块策略
2. 使用 tiktoken (cl100k_base) 替代估算，实现精确 token 计量
3. 引入 10-15% 句子级 overlap，缓解跨 chunk 指代丢失

## 备选方案
- LangChain RecursiveCharacterTextSplitter: 不支持自定义句子正则，overlap 可能断词
- LlamaIndex SentenceSplitter: 依赖 nltk，中文支持差
- 自研 BPE 分块: 开发成本高，且与上游模型 tokenizer 不一致

## 理由
- 句子级切分保证语义完整性，显著提升检索准确率
- tiktoken 是事实标准，与主流 LLM API 计费一致
- 三层防御覆盖所有边界情况，生产环境鲁棒性强

## 后果
- (+) 检索质量提升，Token 成本可控，无上下文溢出风险
- (-) 分块逻辑复杂度增加，需充分测试边界条件
- (!) 当上游模型更换 tokenizer 时，仅需修改 TokenCounter 的 encoding_name