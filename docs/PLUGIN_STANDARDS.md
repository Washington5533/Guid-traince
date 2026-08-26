# DSH Community Plugin Standards

## 概述

本文档定义 DSH 社区插件的标准格式和行为契约。所有通过 `dsh plugin add` 安装到 DSH profile 的插件都应遵守本文档。

---

## 1. 包结构

```
my-dsh-plugin/
├── package.json              # 必须有 `dsh` 段
├── cordis.patch.yml          # 插件 manifest（位置固定）
├── README.md                 # 英文说明
├── README.zh.md              # 中文说明（推荐）
├── tsconfig.json             # TypeScript 配置
├── tsdown.config.ts          # 构建配置（双目标）
├── src/
│   ├── index.ts              # Host half — cordis 入口
│   └── client/
│       ├── index.ts          # Browser half — React UI 入口
│       ├── slots-augment.ts  # SlotMap 类型声明（如有 slot 注入）
│       ├── locales.ts        # i18n 字典
│       ├── panel/            # React 组件
│       ├── settings/         # 设置卡片
│       └── sse/              # SSE 客户端
└── tests/
    └── *.test.ts             # 单元测试
```

---

## 2. `package.json` `dsh` 段

```json
{
  "dsh": {
    "engines": {
      "dsh": ">=0.1.1-rc.1"
    },
    "bundle": {
      "patch": "./cordis.patch.yml"
    },
    "client": {
      "inject": [
        "@deepseek-ai/dsh-client-runtime",
        "@deepseek-ai/dsh-client-connection",
        "@deepseek-ai/dsh-client-locale",
        "@deepseek-ai/dsh-client-ui-settings",
        "@deepseek-ai/dsh-client-ui-slots"
      ],
      "platform": "web"
    }
  }
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `engines.dsh` | ✅ | 最低 DSH 版本 |
| `bundle.patch` | ✅ | `cordis.patch.yml` 路径 |
| `client.inject` | 视情况 | 浏览器端依赖的 DSH runtime bundles |
| `client.platform` | 视情况 | `"web"` 或 `"tui"` |

---

## 3. `cordis.patch.yml` 标准

```yaml
- insert:
    - id: my-plugin                      # 唯一 ID（kebab-case）
      name: '@scope/my-dsh-plugin'        # npm 包名，必须匹配 package.json

  meta:                                  # 社区元数据（可选但推荐）
    version: "1.0.0"                     # 插件版本
    description: "What this plugin does"
    author: "Your Name"
    license: "Apache-2.0"
    homepage: "https://github.com/you/my-dsh-plugin"
    keywords: ["keyword1", "keyword2"]

    # Skill 声明（可选）— 插件向 DSH skill 系统注册的能力
    skills:
      - id: my-skill
        name: "My Skill"
        description: "One-line description"
        whenToUse: "When to invoke this skill"
```

### `meta` 字段规范

| 字段 | 类型 | 说明 |
|------|------|------|
| `version` | string | 语义化版本，必须与 package.json 一致 |
| `description` | string | ≤120 字符，英文 |
| `author` | string | 维护者名称 |
| `license` | string | SPDX 标识符 |
| `homepage` | string | 项目 URL |
| `keywords` | string[] | 搜索标签 |
| `skills` | array | Skill 声明列表 |

---

## 4. Skill 声明规范

Skill 是插件向 DSH 注册的 **能力入口**。DSH 的 skill 系统（`ctx.skills`）负责发现和调度。

```yaml
skills:
  - id: training-guardian           # kebab-case，全局唯一
    name: "Training Guardian"        # 人类可读名称
    description: "Monitor training"  # ≤100 字符
    whenToUse: >                     # 触发条件（给 Agent 的决策依据）
      Use when the user asks about
      training status, GPU, or anomalies.
```

### 浏览器侧 Skill 注册

在 `src/client/index.ts` 的 `apply()` 中注册：

```ts
// 在 apply() 末尾
ctx.skills?.register('training-guardian', {
  name: 'training-guardian',
  description: 'Monitor training jobs in real-time',
  whenToUse: 'Use when the user asks about training status, GPU utilization, or anomalies',
  invoke: () => {
    // 打开面板的逻辑
    const btn = document.querySelector('[data-slot="training-guardian-action"]')
    btn?.dispatchEvent(new MouseEvent('click'))
  },
})
```

---

## 5. Slot 使用规范

### Slot 声明（`slots-augment.ts`）

```ts
declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface SlotMap {
    'conversation.session.header.actions': { kind: 'list', scope: 'root', owner: never }
    'settings.plugin.item':               { kind: 'list', scope: 'root', owner: never }
  }
}
```

### Slot 注入

```ts
ctx.slots.inject('conversation.session.header.actions', () =>
  ctx.slots.register({
    name: 'conversation.session.header.actions',
    id: 'my-plugin',
    order: 10,
  }, () => createElement(MyComponent, props))
)
```

### 推荐的 Slot 列表

| Slot | 用途 | kind |
|------|------|------|
| `conversation.session.header.actions` | 会话头部按钮 | list |
| `settings.plugin.item` | 设置页面插件卡片 | list (keyed) |
| `sidebar.*` | 侧边栏面板 | single |

---

## 6. 构建产物要求

插件必须使用 **双目标构建**（tsdown 双 config 或等价物），发布到 npm 时必须包含：

```
dist/
  index.mjs          # Host half ESM（cordis/Node 入口）
  index.d.mts        # Host half 类型
  index.d.mts.map
dist-client/
  client/
    index.js         # Browser half CJS
    index.d.ts       # Browser half 类型
    index.js.map
```

### 强制要求

1. **`__ModuleLoader__.load` wrapper**：Browser half 的 `index.js` 必须由
   `window.__ModuleLoader__.load({ id, factory })` 包裹（DSH loader 通过
   `./client` export 解析 bundle 时依赖它）。tsdown 用 `banner`/`footer`
   选项实现，参考 `dsh-plugin/dsh-client-ui-training-guardian/tsdown.config.ts`。
2. **文件名必须是 `.js`**：package.json 声明 `"type": "module"` 时 tsdown 默认
   输出 `.cjs`，必须用 `entryFileNames: '[name].js'` 强制 `.js`，否则 DSH
   loader 找不到 `dist-client/client/index.js`。
3. **`./client` export 必须存在**：package.json `exports["./client"].default`
   指向 `./dist-client/client/index.js`。

### 验证命令

```bash
pnpm build
ls dist/index.mjs dist-client/client/index.js                    # 必须存在
grep -q "__ModuleLoader__.load" dist-client/client/index.js      # wrapper 必须存在
# 或用 CLI 一键校验：
python scripts/dsh_plugin_cli.py validate --plugin-dir <plugin-dir>
```

---

## 7. 测试要求

```bash
# 最低要求：vitest 单元测试通过
pnpm test

# 推荐：包含 jsdom 集成测试
pnpm test:integration
```

集成测试应验证：
1. `apply()` 在 mock ctx 上正常调用
2. Slot 注入不抛异常
3. i18n 字典完整性

---

## 8. 自动化发布

插件仓库必须包含 `.github/workflows/plugin-publish.yml`。触发方式：推送
`vX.Y.Z` tag（**必须与 package.json `version` 完全一致**，CI 会校验）。

### 发布流水线

```
git tag v0.2.0 && git push origin v0.2.0
        │
        ▼
  ┌─ validate ────────┐   ① CLI 校验 PLUGIN_STANDARDS 清单
  │                   │   ② tag 版本号 == package.json 版本号
  ├─ test ────────────┤   pnpm typecheck + pnpm test
  ├─ build ───────────┤   pnpm build + 产物校验（§6）+ artifact 上传
  ├─ publish-npm ─────┤   npm publish --access public --provenance
  ├─ update-registry ─┤   重算 registry 条目并提交回默认分支
  └─ github-release ──┘   softprops/action-gh-release 创建 Release
```

要点：

- `pnpm/action-setup@v4` 必须显式 `version:`（与插件 `packageManager` 一致，当前 `11.22.0`）：
  流水线 job 在仓库根目录运行，根目录没有 `package.json`，action 无法自动解析插件目录里的 `packageManager`；
  `setup-node` 同时设 `cache: pnpm`，并把 `cache-dependency-path` 指向插件的 `pnpm-lock.yaml`。
- `publish-npm` / `update-registry` / `github-release` 仅在 tag 触发时运行；
  `workflow_dispatch` 手动触发只跑校验/测试/构建（安全预检）。
- `update-registry` 调用 `scripts/update_registry.py --write`（PyYAML 解析
  cordis.patch.yml，提取 meta + skills + slots），提交消息带 `[skip ci]`
  防止流水线自触发。
- npm 发布需要 `secrets.NPM_TOKEN`；`--provenance` 需要 workflow 级
  `permissions: id-token: write`。

### 发布操作（开发者）

```bash
# 自动 bump 版本（package.json + cordis.patch.yml meta 同步）、
# 本地 registry 同步、commit + tag，--push 直接推送到远端触发 CI：
dsh-plugin release --bump patch --push

# 或手动指定版本 / 仅改文件不打 tag：
dsh-plugin release --version 0.2.0
dsh-plugin release --bump minor --no-git
```

---

## 9. 插件 Registry

社区插件列表托管在 `registry/plugins.json`，schema：

```json
{
  "version": 1,
  "registry_url": "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/registry/plugins.json",
  "updated_at": "2026-08-24",
  "plugins": [
    {
      "name": "@scope/plugin-name",
      "version": "1.0.0",
      "description": "...",
      "author": "Name",
      "license": "Apache-2.0",
      "homepage": "https://github.com/scope/repo",
      "npm": "@scope/plugin-name",
      "source": "https://github.com/scope/repo/tree/main/dsh-plugin/plugin-name",
      "keywords": ["dsh-plugin"],
      "dsh": ">=0.1.1-rc.1",
      "platform": "web",
      "slots": ["conversation.session.header.actions", "settings.plugin.item"],
      "skills": [
        {
          "id": "my-skill",
          "name": "My Skill",
          "description": "...",
          "whenToUse": "..."
        }
      ],
      "updated_at": "2026-08-24"
    }
  ]
}
```

字段来源（由 `scripts/update_registry.py` 自动计算，勿手写）：

| 字段 | 来源 |
|------|------|
| `name`/`npm`/`version`/`description`/`author`/`license`/`homepage`/`keywords` | `package.json`（`meta` 兜底） |
| `dsh` | `package.json` → `dsh.engines.dsh` |
| `platform` | `package.json` → `dsh.client.platform` |
| `slots` | 扫描 `src/client/**` 中 `ctx.slots.inject()` 调用点 |
| `skills` | `cordis.patch.yml` → `meta.skills`（完整对象） |
| `source` | CI 传 `--source-url`（tag 触发时自动生成） |

### 注册流程

1. 插件作者在 repo 根目录维护 `registry/plugins.json`（`dsh-plugin scaffold`
   会自动初始化并写入条目）
2. GitHub Actions 在 tag 发布成功后调用 `update_registry.py --write`
   重算条目并 commit 回默认分支
3. 中央 registry（可 fork 本仓库）聚合所有已知插件
4. `dsh-plugin search <query>` 查询 registry（优先读取本地缓存）

本地手动同步：

```bash
python scripts/update_registry.py --plugin-dir dsh-plugin/<name>            # dry run
python scripts/update_registry.py --plugin-dir dsh-plugin/<name> --write    # 写入
# 或
dsh-plugin registry update --write
```

---

## 10. 安全要求

| 项目 | 要求 |
|------|------|
| Auth Token | SSE 通过 `?token=` query param 传递，REST 通过 `X-Auth-Token` header |
| CORS | 插件必须声明允许的 origin 列表 |
| 输入校验 | 所有用户输入（settings 字段）做类型校验 |
| 依赖 | 最小化，peerDependencies 仅 React |

---

## 11. 开发工具链（dsh-plugin CLI）

本仓库 `scripts/dsh_plugin_cli.py`（pyproject 已注册 console script `dsh-plugin`）
提供插件全生命周期的命令：

### 开发侧

```bash
# 生成一个完全符合本规范的插件骨架（含双目标构建、工作流、registry 条目）
dsh-plugin scaffold my-cool-plugin --author "Your Name" --description "..."

# 运行规范清单校验（§附录，硬性失败返回非零）
dsh-plugin validate --plugin-dir dsh-plugin/my-cool-plugin

# 发布：bump 版本 → 同步 cordis.patch.yml → 同步本地 registry → commit + tag
dsh-plugin release --bump patch --push

# Registry 维护
dsh-plugin registry update --write            # 重算并写入当前插件条目
python scripts/update_registry.py --write     # 等价底层脚本（CI 同款）
```

### 消费侧

```bash
dsh-plugin search training                     # 查询社区 registry
dsh-plugin add @scope/my-plugin --profile web  # 安装（委托 dsh plugin）
dsh-plugin list --profile web                  # 列出已安装插件
dsh-plugin info @scope/my-plugin --profile web # 详情 + 构建产物检查
dsh-plugin remove @scope/my-plugin --profile web
```

---

## 附录：快速 Checklist

```
[ ] package.json 有 `dsh` 段（engines.dsh / bundle.patch / client.platform）
[ ] cordis.patch.yml 存在且 `- insert:` 的 id（kebab-case）+ name 与 package.json 一致
[ ] meta 块存在：version == package.json version，description ≤ 120 字符，author/license 齐备
[ ] meta.skills 声明了插件能力（id/name/description/whenToUse）
[ ] README.md + README.zh.md
[ ] tsconfig.json + tsdown.config.ts 双目标构建
[ ] dist/ 和 dist-client/ 在 .gitignore 中，但 npm pack 包含
[ ] client bundle 有 __ModuleLoader__.load wrapper 且为 .js（非 .cjs）
[ ] exports["./client"] 指向 dist-client/client/index.js
[ ] tests/ 目录有 vitest 测试
[ ] slots-augment.ts 声明了所有 inject() 用到的 slot
[ ] .github/workflows/plugin-publish.yml 存在
[ ] registry/plugins.json 包含本插件（dsh-plugin scaffold / registry update 维护）
[ ] 发布 tag vX.Y.Z == package.json version（dsh-plugin release 自动保证）

一键校验：dsh-plugin validate --plugin-dir <plugin-dir>
```
