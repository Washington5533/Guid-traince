/**
 * Host half — runs inside the DSH cordis host process.
 *
 * For Training Guardian, the host side is minimal: all the real work happens in
 * the browser half (SSE → React panel). This file exists so that cordis can load
 * the plugin; the apply() body registers nothing because we don't need any host-side
 * routes or services — the browser half talks directly to the guarftrain RemoteServer
 * over HTTP/SSE.
 */
export declare function apply(_ctx: never): void;
//# sourceMappingURL=index.d.ts.map