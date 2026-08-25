import { chromium } from '@playwright/test'

async function main() {
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] })
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } })

  console.log('Navigating to DSH web...')
  await page.goto('http://127.0.0.1:3080', { waitUntil: 'networkidle', timeout: 15000 })
  await page.screenshot({ path: 'C:/Users/wst/Desktop/anytries/guarftrain/dsh-web-1-home.png', fullPage: true })
  console.log('Screenshot 1: home page saved')

  // Click first workspace
  console.log('Clicking first workspace...')
  const firstWorkspace = page.locator('.workspace-item, [class*="workspace"]').first()
  if (await firstWorkspace.count() > 0) {
    await firstWorkspace.click()
    await page.waitForTimeout(2000)
    await page.screenshot({ path: 'C:/Users/wst/Desktop/anytries/guarftrain/dsh-web-2-session.png', fullPage: true })
    console.log('Screenshot 2: session view saved')
  } else {
    console.log('No workspace found, trying alternative selector...')
    // Try clicking any clickable item in sidebar
    const sidebarItems = page.locator('text=/chain|ai_interview|anytries|min_prompt|wst|anycpp/').first()
    if (await sidebarItems.count() > 0) {
      await sidebarItems.click()
      await page.waitForTimeout(2000)
      await page.screenshot({ path: 'C:/Users/wst/Desktop/anytries/guarftrain/dsh-web-2-session.png', fullPage: true })
      console.log('Screenshot 2: session view saved')
    }
  }

  // Look for Training Guardian button
  console.log('Looking for Training Guardian button...')
  const tgButton = page.locator('text=Training Guardian, text=训练守护, [aria-label*="Guardian"]').first()
  const tgFound = await tgButton.count() > 0
  console.log(`Training Guardian button found: ${tgFound}`)

  if (tgFound) {
    await tgButton.screenshot({ path: 'C:/Users/wst/Desktop/anytries/guarftrain/dsh-web-3-tg-button.png' })
    await tgButton.click()
    await page.waitForTimeout(1000)
    await page.screenshot({ path: 'C:/Users/wst/Desktop/anytries/guarftrain/dsh-web-4-tg-panel.png', fullPage: true })
    console.log('Screenshot 3-4: TG button and panel saved')
  }

  // Check page content for plugin indicators
  const content = await page.content()
  const hasTrainingGuardian = content.includes('training-guardian') || content.includes('Training Guardian') || content.includes('训练守护')
  console.log(`Page contains TG references: ${hasTrainingGuardian}`)

  // Log all button texts for debugging
  const buttons = await page.locator('button').allTextContents()
  console.log('All buttons:', buttons.slice(0, 20))

  await browser.close()
  console.log('Done!')
}

main().catch(err => { console.error(err); process.exit(1) })
