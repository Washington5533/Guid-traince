import type { SettingsScope } from '@deepseek-ai/dsh-client-runtime/client';
export interface TrainingGuardianSettings {
    serverUrl: string;
    authToken: string;
    sessionId: string;
    autoConnect: boolean;
}
export declare const DEFAULT_SETTINGS: TrainingGuardianSettings;
export type TgSettingsCardFace = {
    onChange(settings: Partial<TrainingGuardianSettings>): void;
};
export declare class TgSettingsCardController {
    private scope;
    private listeners;
    constructor(scope: SettingsScope<TrainingGuardianSettings>);
    getSnapshot(): TrainingGuardianSettings;
    subscribe(fn: () => void): () => void;
    update(partial: Partial<TrainingGuardianSettings>): void;
    dispose(): void;
    inject(): ReturnType<typeof import('react').createElement>;
}
interface SettingsCardProps {
    controller: TgSettingsCardController;
    t: (key: string) => string;
}
export declare function SettingsCard({ controller, t }: SettingsCardProps): any;
export {};
//# sourceMappingURL=SettingsCard.d.ts.map