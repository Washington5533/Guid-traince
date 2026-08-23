import type { SseClient } from '../sse/client';
import type { TgKey } from '../locales';
export interface TrainingPanelProps {
    sse: SseClient;
    sessionId: string | null;
    t: (key: TgKey) => string;
    onApprove: (actionId: string) => void;
    onReject: (actionId: string, reason: string) => void;
}
export declare function TrainingPanel({ sse, sessionId, t, onApprove, onReject }: TrainingPanelProps): any;
//# sourceMappingURL=TrainingPanel.d.ts.map