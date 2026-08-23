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
      inject<K extends string>(seat: K, spec: SlotSpec): void
      register<K extends string>(seat: K, spec: SlotSpec): void
    }
    webUiSettings?: {
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

type SlotSpec = {
  kind: 'single' | 'list'
  inject: () => Promise<{
    default: (props: Record<string, unknown>) => unknown
  }>
}
