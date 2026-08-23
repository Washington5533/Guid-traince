/**
 * DSH browser-side entry for Training Guardian.
 *
 * Registered by the host half (src/index.ts) as a `dsh.client` entry.
 * Reads settings from the DSH settings scope, constructs the SSE client,
 * and injects the sidebar panel + settings card into DSH slots.
 */
import type { Context } from '@deepseek-ai/dsh-client-runtime';
export interface TgClientOptions {
    ctx: Context;
    localeNs?: string;
}
export declare function apply({ ctx, localeNs }: TgClientOptions): void;
//# sourceMappingURL=index.d.ts.map