import type { TgKey } from '../locales';
interface DecisionsTabProps {
    pending: Array<Record<string, unknown>>;
    t: (key: TgKey) => string;
    onApprove: (actionId: string) => void;
    onReject: (actionId: string, reason: string) => void;
}
export declare function DecisionsTab({ pending, t, onApprove, onReject }: DecisionsTabProps): any;
export {};
//# sourceMappingURL=DecisionsTab.d.ts.map