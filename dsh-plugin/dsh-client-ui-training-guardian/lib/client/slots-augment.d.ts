/**
 * Type augmentation for DSH SDK slot declarations.
 *
 * These module augmentations merge our slot types into the DSH SDK's
 * SlotMap so that ctx.slots.inject() / ctx.slots.register() type-check.
 */
declare module '@deepseek-ai/dsh-client-ui-slots' {
    interface SlotMap {
        /**
         * Sidebar panel seat for the Training Guardian panel.
         * The DSH sidebar shell declares this seat for plugin panels.
         */
        'sidebar.training-guardian': {
            kind: 'single';
            scope: 'root';
            owner: never;
        };
        /**
         * Plugin item card in the Web UI plugin group (settings → plugins).
         */
        'web-ui.plugin.item': {
            kind: 'list';
            scope: 'root';
            owner: never;
        };
    }
}
declare module '@deepseek-ai/dsh-client-runtime' {
    interface Context {
        /**
         * Optional compatibility binder from dsh-web-ui-settings (rc.6 compat).
         */
        webUiSettings?: {
            bind<S>(spec: SettingsScopeSpec<S>): SettingsScope<S>;
        };
    }
}
//# sourceMappingURL=slots-augment.d.ts.map