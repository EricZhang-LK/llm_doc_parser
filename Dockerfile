# Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装 uv 工具
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 复制项目依赖文件
COPY pyproject.toml uv.lock ./

# 使用 uv 同步依赖（只安装生产环境依赖）
RUN uv sync --frozen --no-dev

# 复制源代码（含 src/llm_doc_parser/prompts/ 模板）
COPY src/ ./src/

# 设置环境变量
ENV PYTHONPATH=/app/src

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uv", "run", "python", "-m", "llm_doc_parser.api.main"]