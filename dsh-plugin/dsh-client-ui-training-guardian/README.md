# dsh-client-ui-training-guardian

DSH Web GUI plugin that adds a **Training Guardian** sidebar panel with
real-time training metrics, GPU device status, anomaly feed, and a sub-agent
decision approval workflow.

It consumes the SSE + REST API exposed by [`guarftrain`](https://github.com/Washington5533/guarftrain)'s
`guardian.remote.RemoteServer`, so it works with any training job that has a
guardian server running alongside it.

## Features

| Tab | Content |
|-----|---------|
| **Overview** | Epoch, step, loss, accuracy, learning rate, status |
| **Devices** | Per-GPU utilization, temperature, VRAM, power with mini progress bars |
| **Anomalies** | Live feed of anomaly events with severity badges |
| **Decisions** | Sub-agent proposed actions with Approve / Reject inline |

A **Settings** card (in the DSH plugin settings area) lets users configure the
guardian server URL, auth token, and session ID, with optional auto-connect.

## Installation

```bash
# Inside your DSH Web GUI deployment:
cd dsh-web-ui/plugins
pnpm add @linxin666/dsh-client-ui-training-guardian
```

Restart the DSH host. The plugin is auto-discovered via its `cordis.patch.yml`.

## Configuration

Open **Settings → Plugins → Training Guardian** and fill in:

- **Guardian Server URL** — e.g. `http://192.168.1.100:8765` (the `RemoteServer` host).
- **Auth Token** — optional; must match the `--remote-auth` flag if the server requires it.
- **Session ID** — optional; leave blank to auto-subscribe to the first active training session.
- **Auto Connect** — connect automatically when the panel opens.

## Requirements

- DSH Web GUI `>= 0.1.1-rc.1`
- Node.js `^22.19.0 || >=24.0.0`
- A running `guarftrain` guardian server with `RemoteServer` enabled (`--remote`).

## Development

```bash
pnpm install
pnpm typecheck   # tsc -b
pnpm build       # tsc -b && tsdown
```

## License

Apache-2.0
