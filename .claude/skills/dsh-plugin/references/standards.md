# PLUGIN_STANDARDS 速查（精缩版）

> 规范原文：`docs/PLUGIN_STANDARDS.md`。本文是执行校验/评审时的快速清单。

## 包结构

```
my-dsh-plugin/
├── package.json              # 必须有 `dsh` 段
├── cordis.patch.yml          # 插件 manifest（位置固定）
├── README.md / README.zh.md
├── tsconfig.json / tsdown.config.ts   # 双目标构建
├── src/
│   ├── index.ts              # Host half — cordis 入口
│   └── client/
│       ├── index.ts          # Browser half — apply(ctx) 入口
│       ├── slots-augment.ts  # SlotMap 类型声明
│       ├── locales.ts        # i18n 字典
│       ├── panel/ settings/ sse/
└── tests/*.test.ts           # vitest
```

## package.json `dsh` 段（必填）

```json
"dsh": {
  "engines": { "dsh": ">=0.1.1-rc.1" },
  "bundle": { "patch": "./cordis.patch.yml" },
  "client": {
    "inject": ["@deepseek-ai/dsh-client-runtime", "@deepseek-ai/dsh-client-connection",
               "@deepseek-ai/dsh-client-locale", "@deepseek-ai/dsh-client-ui-settings",
               "@deepseek-ai/dsh-client-ui-slots"],
    "platform": "web"
  }
}
```

## cordis.patch.yml 硬性要求

- 顶层 list，含 `- insert:` 块：`id`（kebab-case 全局唯一）+ `name`（== package.json name）
- `meta:` 块：`version` == package.json version；`description` ≤ 120 字符；
  `author`、`license`（SPDX）、`homepage`、`keywords`
- `meta.skills`：`[{id, name, description, whenToUse}]`

## 构建产物硬性要求（§6）

- `dist/index.mjs`（host ESM）+ `dist/index.d.mts`
- `dist-client/client/index.js`（client CJS）+ `index.d.ts`
- client bundle 必须以 `window.__ModuleLoader__.load({ id, factory })` 包裹
- 文件后缀必须 `.js`（不是 `.cjs`）→ `entryFileNames: '[name].js'`
- `exports["./client"].default` → `./dist-client/client/index.js`

## Slot 规范

- 声明：`slots-augment.ts` 里 `declare module '@deepseek-ai/dsh-client-ui-slots'`
  的 `interface SlotMap`（每个 inject 的 slot 必须有条目）
- 注入：`ctx.slots.inject(name, () => ctx.slots.register(opts, Component))`，
  必须包 try/catch 优雅降级
- 推荐 slot：`conversation.session.header.actions`（list）、`settings.plugin.item`（keyed）

## 测试最低要求

- `tests/` 有 vitest 测试；集成测试验证 `apply()` 在 mock ctx 上不抛异常、
  slot 注入不抛异常、i18n 字典完整

## 发布规则

- tag `vX.Y.Z` == package.json version（硬性）
- 用 `dsh-plugin release --bump <patch|minor|major> [--push]` 完成
  bump + meta 同步 + registry 同步 + commit + tag
- workflow：validate → test/build → publish-npm（--provenance）→
  update-registry（update_registry.py，`[skip ci]` commit）→ github-release
- 需要 secrets：`NPM_TOKEN`；permissions：`contents: write` + `id-token: write`

## 校验清单（validate 命令覆盖）

1. `dsh` 段完整（engines.dsh / bundle.patch / client.platform ∈ {web, tui}）
2. insert id kebab-case；insert name == package.json name
3. meta 块存在；meta.version == package.json version；description ≤ 120
4. skills 每个条目有 id/name/description（缺失 skills 为 ⚠ 建议）
5. README.md（硬性）+ README.zh.md（建议）
6. tsconfig.json + tsdown.config.ts
7. tests/*.test.ts ≥ 1
8. dist/index.mjs + dist-client/client/index.js 存在（⚠，未构建时）
9. client bundle 含 `__ModuleLoader__.load`（⚠）
10. slots-augment.ts 覆盖全部 inject() 调用点
11. repo 有 `.github/workflows/plugin-publish.yml`
12. registry/plugins.json 含本插件（⚠）
