# ADR-002: 选择 Docling 作为默认文档解析引擎

## 状态
已采纳 (2026-06-03)

## 背景
RAG 系统的检索质量高度依赖文档解析精度。需要支持 PDF 中的表格、公式、
多栏排版等复杂结构，同时提供干净的 Python API 以适配生产级异步服务。

## 决策
采用 IBM Docling v2 作为默认解析引擎，通过适配器模式封装。

## 备选方案
- MinerU (Magic-PDF): 中文支持好，但依赖重、API 不够 Pythonic
- LlamaIndex SimpleDirectoryReader: 功能太弱，不支持表格/公式
- Unstructured: 商业版贵，开源版功能受限

## 理由
- 原生多模态支持（文本+表格+图片+公式）
- 干净的 Python API，易于适配异步和 Pydantic 契约
- IBM 维护活跃，社区增长快
- 模型权重可离线使用，适合私有化部署

## 后果
- (+) 解析质量显著优于传统方案，API 友好
- (-) 依赖 torch，镜像体积增加约 2GB；首次加载模型较慢
- (!) 通过适配器隔离，未来可平滑切换至 MinerU 或自研方案