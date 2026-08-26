---
name: dsh-plugin
description: Develop, validate, scaffold, and release DSH community plugins (dsh-client-ui-* npm packages with a `dsh` section, cordis.patch.yml, and dual-target tsdown build). Use when creating a new DSH plugin, checking one against PLUGIN_STANDARDS, cutting a release tag that triggers the publish pipeline, or updating registry/plugins.json.
---

# DSH 社区插件开发与发布

DSH 社区插件 = npm 包（含 `dsh` 段 + `cordis.patch.yml`）+ 双目标构建产物
（host ESM → `dist/`，client CJS + `__ModuleLoader__` wrapper → `dist-client/`），
通过 `git tag vX.Y.Z` 触发 GitHub Actions 全自动发布（npm → registry → Release）。

**规范原文（canonical spec）**：`docs/PLUGIN_STANDARDS.md` — 任何约定冲突时以它为准。
**快速参考**：`references/standards.md`（清单速查）、`references/registry-schema.md`（registry schema）。

## 本仓库结构

| 位置 | 内容 |
|------|------|
| `dsh-plugin/dsh-client-ui-training-guardian/` | 参考插件（Training Guardian，本规范的实现范例） |
| `docs/PLUGIN_STANDARDS.md` | 社区插件规范 |
| `scripts/dsh_plugin_cli.py` | 插件 CLI（`dsh-plugin`，pyproject 已注册） |
| `scripts/update_registry.py` | registry 条目重算器（CI 与本地共用） |
| `.github/workflows/plugin-publish.yml` | 发布流水线 |
| `registry/plugins.json` | 社区插件 registry |

## 工作流 1：脚手架（新建插件）

```bash
python scripts/dsh_plugin_cli.py scaffold my-cool-plugin \
  --author "Your Name" --description "What it does" [--platform web|tui]
```

- 在仓库内运行：生成 `dsh-plugin/my-cool-plugin/`，并自动向 `registry/plugins.json`
  写入条目；若仓库缺 `scripts/` 工具链会自动复制。
- 在全新目录运行（`--dir /path/new-repo`）：生成完整仓库布局
  （`dsh-plugin/<slug>/` + `scripts/` + `registry/`）。
- 生成的骨架已含：package.json（`dsh` 段）、cordis.patch.yml（insert + meta +
  skills）、双目标 tsdown.config.ts（含 `__ModuleLoader__` wrapper）、
  slot 注入 + skill 注册的 client 入口、locales、slots-augment、vitest 冒烟测试、
  双语 README、`.github/workflows/plugin-publish.yml`。
- 名字规范：npm 包名 `@scope/kebab-name` 或 `kebab-name`；insert `id` 取
  kebab-name（全局唯一）。

## 工作流 2：开发迭代

```bash
cd dsh-plugin/<plugin>
pnpm install
pnpm build          # tsc --noEmit && tsdown（产出 dist/ + dist-client/）
pnpm test           # vitest
pnpm typecheck
```

改完源码后（本地联调）：`pnpm build`，dsh-wsl dev 模式会自动 reload client bundle。

### 实时联调（真机验证）

```bash
# 1. DSH web（WSL，用 nvm node + corepack pnpm；Windows PATH 里的 pnpm 在 WSL 不可用）
#    wsl -e bash -c '. ~/.nvm/nvm.sh; nvm use 22; cd ~/dsh-wsl && nohup pnpm dsh web &'
# 2. 真训练 + 远程服务（SSE/REST 数据源）
python -m guardian.cli watch --remote --remote-port 8765 -- python scripts/train.py \
  --epochs 20 --ckpt_dir ./checkpoints_live --data_dir ./data
# 3. 浏览器自动化验证（.tmp-playwright/，playwright-core + 系统 chromium）
#    node .tmp-playwright/live-test.js   # 按钮/面板/SSE 指标/REST 全链路
#    node .tmp-playwright/test-drag.js   # 面板拖拽/吸附/位置持久化
```

验证点：会话头按钮渲染、面板 5 个 tab、SSE 指标实时刷新、异常事件、
REST 断线补拉（`/sse` 连接后回放最近 50 条持久化事件）、面板拖拽与边缘吸附。

## 工作流 3：规范校验

```bash
python scripts/dsh_plugin_cli.py validate --plugin-dir dsh-plugin/<plugin>
```

硬性失败（✗，退出码 1）必须修复；⚠ 为建议项。校验覆盖：`dsh` 段完整性、
insert id/name 与 package.json 一致、meta.version 同步、skills 完整性、
`./client` export、双目标构建产物 + `__ModuleLoader__` wrapper、
slots-augment 与 inject() 调用点一致性、repo 级工作流与 registry 条目。

## 工作流 4：发布（触发自动化部署）

```bash
python scripts/dsh_plugin_cli.py release --bump patch --push
```

`release` 做的事：dirty-tree 检查 → bump package.json `version` → 同步
cordis.patch.yml `meta.version` → 同步本地 registry 条目 → commit
`chore(release): <pkg> vX.Y.Z` → `git tag vX.Y.Z` →（--push）推送触发 CI。

CI 流水线（`.github/workflows/plugin-publish.yml`）：
`validate`（标准校验 + tag==version）→ `test` / `build`（产物校验 + artifact）
→ `publish-npm`（`npm publish --access public --provenance`，需 `NPM_TOKEN`）
→ `update-registry`（`update_registry.py --write` 提交回默认分支，`[skip ci]`）
→ `github-release`。

硬性规则：**tag `vX.Y.Z` 必须与 package.json version 完全一致**，CI 会拒绝不一致。

## 工作流 5：Registry 维护

```bash
python scripts/update_registry.py --plugin-dir dsh-plugin/<plugin>          # dry run
python scripts/update_registry.py --plugin-dir dsh-plugin/<plugin> --write  # 写入
# 等价：python scripts/dsh_plugin_cli.py registry update --write
```

条目由 manifest 自动计算：`package.json`（name/version/description/…/dsh/platform）
+ `cordis.patch.yml` meta（skills 完整对象）+ 源码扫描（slots =
`ctx.slots.inject()` 调用点）。已有条目的手工字段会被保留（merge 而非覆盖），
内容无变化时不会 bump registry 版本号。

## 工作流 6：消费侧操作

```bash
python scripts/dsh_plugin_cli.py search training                    # 查询 registry
python scripts/dsh_plugin_cli.py add <pkg> --profile web            # 安装（委托 dsh plugin）
python scripts/dsh_plugin_cli.py list --profile web                 # 已安装列表
python scripts/dsh_plugin_cli.py info <pkg> --profile web           # 详情 + 产物检查
python scripts/dsh_plugin_cli.py remove <pkg> --profile web
```

## 常见失败模式（务必检查）

1. **client bundle 输出 `.cjs` 而非 `.js`** — package.json `"type": "module"`
   时 tsdown 默认后缀 `.cjs`，必须 `entryFileNames: '[name].js'`。
2. **缺 `__ModuleLoader__.load` wrapper** — DSH loader 无法挂载 client 半体。
3. **`meta.version` 与 package.json 不同步** — 用 `release` 命令改版本，不要手改。
4. **tag 与版本不一致** — CI `validate`/`publish-npm` 双重拦截。
5. **registry 条目 clobber** — 旧 workflow 用正则重写导致 skills 富信息丢失；
   一律走 `update_registry.py`（YAML 解析 + merge 保留）。
6. **pnpm 版本 pin 冲突** — workflow 不要 pin `pnpm/action-setup` version，
   让它读插件 `packageManager` 字段。

## 参考插件对照

改任何规范条目时，同步检查参考插件 `dsh-plugin/dsh-client-ui-training-guardian/`
是否仍满足，并跑一遍 `validate`。
