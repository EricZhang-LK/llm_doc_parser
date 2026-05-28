# project-template

Python project template with CI, lint, type checks, tests, and commit conventions. Designed to be cloned or used as a GitHub template for new projects.

中文：这是一个包含 CI、代码规范检查、类型检查、测试和提交规范的 Python 项目模板，可直接通过 GitHub Template、fork 或 clone 用于新项目初始化。

## What This Template Includes

- Python package layout under `src/`
- Test layout under `tests/`
- CI workflow for lint, format check, type check, and tests
- Pre-commit hooks and optional commit message linting
- ADR folder for architecture decision records

中文：
- 使用 `src/` 目录组织 Python 包
- 使用 `tests/` 目录组织测试代码
- 提供 CI 工作流（lint、格式检查、类型检查、单元测试）
- 提供 pre-commit 与可选的 commitlint 提交规范检查
- 提供 ADR 目录用于记录架构决策

## Quick Start

1. Create and activate a virtual environment.
2. Install project and dev dependencies: `pip install -e ".[dev]"`.
3. Install and enable hooks: `pip install pre-commit && pre-commit install`.
4. (Optional) Install Node hooks toolchain: `npm install`.
5. Run local checks: `ruff check . && ruff format --check . && mypy src && pytest`.

中文：
1. 创建并激活虚拟环境。
2. 安装项目与开发依赖：`pip install -e ".[dev]"`。
3. 安装并启用 hooks：`pip install pre-commit && pre-commit install`。
4. （可选）安装 Node 工具链以启用 husky/commitlint：`npm install`。
5. 运行本地检查：`ruff check . && ruff format --check . && mypy src && pytest`。

## Template Usage

1. Click **Use this template** on GitHub, or fork this repository.
2. Rename package directories in `src/` and update imports in `tests/` if needed.
3. Replace placeholder metadata in `pyproject.toml`, `README.md`, and repository settings.
4. Enable branch protection and required status checks (`CI`).
5. Keep `docs/adr` and add ADR files when major technical decisions are made.

中文：
1. 在 GitHub 点击 **Use this template**，或直接 fork 本仓库。
2. 按需重命名 `src/` 下的包目录，并同步修改 `tests/` 中的导入路径。
3. 替换 `pyproject.toml`、`README.md` 与仓库设置中的占位元信息。
4. 开启分支保护，并将 `CI` 设为必需检查项。
5. 保留 `docs/adr`，在关键技术决策时新增 ADR 文档。

## Suggested First Changes In a New Project

1. Update `project.name`, `description`, and `version` in `pyproject.toml`.
2. Rename the package from `project_template` to your real domain package name.
3. Replace example models and tests with domain-specific logic.
4. Configure release and deployment workflows based on your target environment.

中文：
1. 在 `pyproject.toml` 中更新 `project.name`、`description` 和 `version`。
2. 将示例包名 `project_template` 改为你的真实业务包名。
3. 用你的业务逻辑替换示例模型和测试。
4. 按目标环境补充发布与部署流程。