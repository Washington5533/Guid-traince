# dsh-client-ui-training-guardian

A DSH Web GUI plugin that adds a **Training
Guardian** panel: real-time training metrics, GPU device status, anomaly
feed, sub-agent decision approval, model architecture analysis, and offline
history replay — all inside the DSH sidebar.

It is the frontend companion of
[`guarftrain`](https://github.com/Washington5533/Guid-traince) (Training
Guardian Agent). The panel consumes the SSE + REST API exposed by
`guardian.remote.RemoteServer`, so it works with **any** training job that has
a guardian server running alongside it — your training script stays untouched.

> Standalone plugin mirror: <https://github.com/Washington5533/Guid-traince>

## Features

Six tabs, one connection:

| Tab | Content |
|-----|---------|
| **Overview** | Epoch, step, loss, accuracy, learning rate, training status, and a live dual-axis loss/accuracy chart |
| **Devices** | Per-GPU utilization, temperature, VRAM and power draw with mini progress bars (CPU-only hosts degrade gracefully) |
| **Anomalies** | Live feed of anomaly events with severity badges (loss spike / NaN / OOM risk / stall …) |
| **Decisions** | Sub-agent proposed actions (adjust LR, early stop, rollback checkpoint …) with inline **Approve / Reject** |
| **Architecture** | Model architecture analysis served by the guardian: parameter counts, FLOPs, bottleneck layers |
| **History** | Browse past training sessions — from the guardian server when online, from browser `localStorage` when offline |

Additional capabilities:

- **Session header action** — a Training Guardian button in the conversation
  session header opens/anchors the panel for the current session.
- **Three-state connection** — `idle → connecting → connected` with clear
  error hints (auth failed / server unreachable / retries exhausted) and
  automatic reconnection with backoff.
- **Offline history persistence** — metrics received during a session are
  cached in `localStorage` (up to 2000 points per session, 50 sessions), so
  the History tab keeps working after refresh or when the server is down.
- **i18n** — English and 简体中文 UI strings.
- **Skill integration** — registers a `training-guardian` skill so the agent
  can open the panel when you ask about training status, GPU, loss curves,
  anomalies or decisions.

## Requirements

- DSH Web GUI `>= 0.1.1-rc.1`
- Node.js `^22.19.0 || >=24.0.0`
- A running `guarftrain` guardian server with the remote API enabled (see below)

## Installation

### Option A — from the community registry (recommended)

```bash
dsh plugin add @rrrelink/dsh-client-ui-training-guardian --profile web
```

Or via the `dsh-plugin` helper shipped with guarftrain:

```bash
python scripts/dsh_plugin_cli.py add @rrrelink/dsh-client-ui-training-guardian --profile web
python scripts/dsh_plugin_cli.py validate @rrrelink/dsh-client-ui-training-guardian --profile web
```

### Option B — from source (development)

```bash
git clone https://github.com/Washington5533/Guid-traince.git
cd Guid-traince/dsh-plugin/dsh-client-ui-training-guardian
pnpm install
pnpm build                 # tsc --noEmit && tsdown
dsh plugin add . --profile web   # or copy the built package into ~/.dsh/profiles/web/node_modules/
```

Restart / reload the DSH Web GUI. The plugin is auto-discovered through its
`cordis.patch.yml` and injects into two slots:

- `conversation.session.header.actions` — the panel toggle button
- `settings.plugin.item` — the settings card

## Starting the guardian backend

The plugin only renders data; the data comes from a guardian remote server.
On the training machine:

```bash
pip install guarftrain          # or: pip install . from the repo root

# Start the remote API server (default port 8765, listens on 0.0.0.0)
guarftrain remote --port 8765 --auth <your-token>     # --auth is optional

# In another shell, run training under guardian supervision (zero code change)
guarftrain watch -- python train.py --epochs 50
```

Events are persisted as JSONL under the guardian log directory, which powers
the History API (`/api/history/...`).

## Configuration

Open **Settings → Plugins → Training Guardian** and fill in:

| Field | Default | Description |
|-------|---------|-------------|
| **Server URL** | `http://localhost:8765` | Address of the `RemoteServer` (the training machine) |
| **Auth Token** | *(empty)* | Must match `guarftrain remote --auth` when the server requires a token |
| **Session ID** | *(empty)* | Leave blank to auto-subscribe to the first active training session |
| **Auto Connect** | `on` | Establish the SSE connection automatically when the panel opens |
| **Model Entry** | *(empty)* | e.g. `train:build_model` — used by the Architecture tab |
| **Project Dir** | *(empty)* | Training project root, resolves imports for architecture analysis |
| **Dashboard URL** | *(empty)* | Optional link to the standalone guardian dashboard |

Settings are persisted through the DSH settings scope and take effect on the
next (re)connect — use the **Connect / Disconnect** button in the panel header.

## Usage guide

1. **Connect** — configure the server URL (and token if needed), open the
   panel from the session header, and watch the status badge turn green.
2. **Monitor** — the Overview tab streams `metrics` events in real time; the
   chart plots loss (left axis, blue) and accuracy (right axis, green).
3. **React to anomalies** — the Anomalies tab collects alerts with severity;
   use it together with the guardian rule engine on the server side.
4. **Approve decisions** — when the sub-agent proposes an action (in
   `supervised` autonomy mode), a card appears in the Decisions tab; click
   **Approve** or **Reject**. Approvals are session-scoped.
5. **Inspect history** — the History tab lists past sessions. Online, it reads
   from the server (`GET /api/history/sessions`); offline, it falls back to
   the browser cache. Selecting a session shows its summary, metrics trend,
   anomalies and decisions.

### Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `auth failed` on connect | Token mismatch with `--auth`, or token missing |
| Retries exhausted, `server unreachable` | Server not started, wrong URL/port, or firewall blocking 8765 |
| Panel shows "no active session" | Training not started under `guarftrain watch`, or a finished session ID was configured |
| History tab empty while offline | No cached session yet — cache is built only while connected and receiving metrics |
| Architecture tab shows estimates only | Model forward pass unavailable (dummy input failed); guardian falls back to static estimation by design |

## Development

```bash
pnpm install
pnpm typecheck   # tsc --noEmit
pnpm test        # vitest run (9 suites)
pnpm build       # tsc --noEmit && tsdown
```

Publishing is automated: the GitHub workflow `plugin-publish.yml` runs on a
version tag — it validates the plugin (28 checks), runs tests, builds,
publishes to npm, and updates `registry/plugins.json`.

## License

MIT — see [LICENSE](LICENSE).
