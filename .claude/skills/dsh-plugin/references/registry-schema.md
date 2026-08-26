# registry/plugins.json Schema

```jsonc
{
  "version": 2,                // int, 每次内容变化 +1
  "registry_url": "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/registry/plugins.json",
  "updated_at": "2026-08-24",  // YYYY-MM-DD
  "plugins": [
    {
      "name": "@scope/plugin-name",   // == npm 包名（主键）
      "version": "1.0.0",
      "description": "...",
      "author": "Name",
      "license": "Apache-2.0",        // SPDX
      "homepage": "https://...",
      "npm": "@scope/plugin-name",
      "source": "https://github.com/<owner>/<repo>/tree/<branch>/<plugin-rel-dir>",
      "keywords": ["dsh-plugin"],
      "dsh": ">=0.1.1-rc.1",          // package.json → dsh.engines.dsh
      "platform": "web",              // package.json → dsh.client.platform
      "slots": ["conversation.session.header.actions"],  // 源码 inject() 扫描
      "skills": [                     // cordis.patch.yml → meta.skills（完整对象）
        { "id": "...", "name": "...", "description": "...", "whenToUse": "..." }
      ],
      "updated_at": "2026-08-24"
    }
  ]
}
```

## 写入规则（scripts/update_registry.py）

- 条目按 `name` 去重 upsert，插件列表按 name 排序
- **merge 保留**：已有条目中 manifest 无法提供的字段（手工补的 slots 等）不会被覆盖
- 内容无变化 → 不 bump `version`、不改 `updated_at`、退出码 0
- 有变化 → `version +1`，`updated_at` 置当天
- `meta.version != package.json version` 时打印 warning 并以 package.json 为准
- 依赖 PyYAML（guarftrain 核心依赖；CI 在 venv 里安装）

## 消费方

- `dsh-plugin search <query>` — 匹配 name / description / keywords
- registry 变更在 CI `update-registry` job 里提交（`[skip ci]`）
