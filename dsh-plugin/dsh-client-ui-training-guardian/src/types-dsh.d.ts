/**
 * Ambient declarations for types provided by DSH runtime packages.
 *
 * These packages are injected by the DSH host at runtime and are not
 * available during local type-checking / testing. The declarations here
 * provide just enough shape for the plugin to compile.
 */

declare module '@deepseek-ai/dsh-client-runtime' {
  export interface SettingsScopeSpec<S> {
    default: S
    namespace: string
  }

  export interface SettingsScope<S> {
    getSnapshot(): { status: string; value?: S }
    setValue(value: S): void
  }

  export interface Context {
    locale: {
      register(namespace: string, dict: { zh: Record<string, string>; en: Record<string, string> }): void
      t(key: string): string
    }
    slots: {
      /**
       * Contribute into an existing slot seat. The thunk runs when the seat
       * resolves and returns the value of {@link register}.
       */
      inject<K extends string>(seat: K, thunk: () => unknown): void
      /**
       * Register a component under a seat: `id` for list seats, `key` for
       * keyed seats (e.g. `settings.plugin.item` is keyed by the settings
       * namespace the card edits).
       */
      register<K extends string>(
        options: {
          name: K
          id?: string
          key?: string
          order?: number
          locale?: string
        },
        component: (props: Record<string, unknown>) => unknown,
      ): unknown
    }
    settingsScope?: {
      bind<S>(spec: SettingsScopeSpec<S>): SettingsScope<S>
    }
  }
}

declare module '@deepseek-ai/dsh-client-runtime/client' {
  export type { SettingsScope, SettingsScopeSpec } from '@deepseek-ai/dsh-client-runtime'
}

declare module '@deepseek-ai/dsh-client-ui-slots' {
  export interface SlotMap {
    [key: string]: {
      kind: 'single' | 'list'
      scope: 'root' | 'workspace'
      owner?: string
    }
  }
}

declare module '@deepseek-ai/dsh-client-ui-settings' {
  export type { SettingsScope } from '@deepseek-ai/dsh-client-runtime'
}
