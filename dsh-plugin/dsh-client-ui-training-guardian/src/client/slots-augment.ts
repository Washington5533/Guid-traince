/**
 * Module augmentation for DSH SDK slot declarations.
 *
 * Augments the DSH SlotMap so that ctx.slots.inject() / ctx.slots.register()
 * type-check for our custom seats.
 */

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface SlotMap {
    /**
     * Sidebar panel seat for the Training Guardian panel.
     * The DSH sidebar shell declares this seat for plugin panels.
     */
    'sidebar.training-guardian': {
      kind: 'single'
      scope: 'root'
      owner: never
    }

    /**
     * Plugin item card in the Web UI plugin group (settings → plugins).
     */
    'web-ui.plugin.item': {
      kind: 'list'
      scope: 'root'
      owner: never
    }
  }
}
