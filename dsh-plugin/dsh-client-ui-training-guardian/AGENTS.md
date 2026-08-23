# AGENTS.md — dsh-client-ui-training-guardian

This directory is a self-contained DSH Web GUI plugin. The runtime and build
tooling are Node.js only — Python files in the parent `guarftrain/` tree are
not part of this package.

## Structure

```
src/
  index.ts              ← host half (cordis entry point, no-op)
  client/
    index.ts            ← browser half (apply() – registers slots + settings)
    sse/
      client.ts         ← EventSource client with auto-reconnect
    panel/
      TrainingPanel.tsx ← main panel (tab bar + tab content)
      MetricsTab.tsx    ← epoch / loss / lr / step cards
      GpuTab.tsx        ← per-GPU utilization / temp / VRAM / power
      AnomaliesTab.tsx  ← anomaly event table
      DecisionsTab.tsx  ← approve / reject workflow
    settings/
      SettingsCard.tsx  ← TgSettingsCardController + React form
    locales.ts          ← zh / en i18n dictionary (TgKey union)
    slots-augment.ts    ← module augmentation for DSH slot types
```

## Commands

```bash
pnpm install        # install dependencies
pnpm typecheck      # tsc -b (noEmit)
pnpm build          # tsc -b && tsdown (produces lib/)
pnpm test           # vitest (when configured)
```

## Coding conventions

- Strict TypeScript, no `any`.
- i18n: every user-visible string is a key in `locales.ts`; use the `TgKey`
  union so missing keys are a compile error.
- Styling: inline `style` objects; prefer CSS custom properties when
  referencing theme colours (`var(--accent, #4fc3f7)`).
- DSH slots: always wrap `inject()` calls in `try/catch` so the plugin
  degrades gracefully on older DSH runtimes.
- SSE client: subscribe to named event types (`metrics`, `gpu_status`,
  `anomaly`, `decision`); never mutate payloads before passing to React state.

## Key types

| Type | Location | Purpose |
|------|----------|---------|
| `TgKey` | `locales.ts` | Union of all i18n keys |
| `SseClient` | `sse/client.ts` | EventSource wrapper |
| `TrainingPanelProps` | `panel/TrainingPanel.tsx` | Props for main panel |
| `TrainingGuardianSettings` | `settings/SettingsCard.tsx` | Settings shape |
| `TgSettingsCardController` | `settings/SettingsCard.tsx` | Settings state manager |
