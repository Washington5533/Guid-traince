/**
 * Module augmentation for DSH SDK slot and locale declarations.
 *
 * Augments the DSH SlotMap so that ctx.slots.inject() / ctx.slots.register()
 * type-check for our custom seats.
 *
 * Augments the DSH LocaleNamespaceMap so that the `t` prop is typed
 * when a slot entry declares `locale: 'training-guardian'`.
 */

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface SlotMap {
    /**
     * Session-header actions list. Each registered entry renders as a
     * button in the conversation session header bar.
     */
    'conversation.session.header.actions': {
      kind: 'list'
      scope: 'session'
      owner: never
    }

    /**
     * Plugin item card in the settings → plugins tab.
     * Keyed by the plugin's settings namespace.
     */
    'settings.plugin.item': {
      kind: 'keyed'
      scope: 'root'
      owner: never
    }
  }

  interface LocaleNamespaceMap {
    /** Training Guardian UI strings. */
    'training-guardian': {
      'panel.title': string
      'panel.overview': string
      'panel.devices': string
      'panel.anomalies': string
      'panel.decisions': string
      'panel.architecture': string
      'settings.serverUrl': string
      'settings.authToken': string
      'settings.sessionId': string
      'settings.autoConnect': string
      'settings.serverUrlHint': string
      'settings.authTokenHint': string
      'settings.sessionIdHint': string
      'settings.saved': string
    }
  }
}
