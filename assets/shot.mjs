// Render promo.txt through carbon.now.sh and screenshot the export frame.
import { chromium } from 'playwright-core';
import { readFileSync } from 'fs';

const code = readFileSync(new URL('./promo.txt', import.meta.url), 'utf8').trimEnd();
const params = new URLSearchParams({
  bg: 'rgba(0,0,0,0)',
  t: 'dracula-pro',
  wt: 'none',
  l: 'application/x-sh',
  width: '680',
  ds: 'true',
  dsyoff: '14px',
  dsblur: '48px',
  wc: 'true',
  wa: 'true',
  pv: '96px',
  ph: '64px',
  ln: 'false',
  fm: 'JetBrains Mono',
  fs: '13px',
  lh: '152%',
  si: 'false',
  es: '2x',
  wm: 'false',
  code,
});

const browser = await chromium.launch({
  executablePath: process.env.CHROME,
  args: ['--no-sandbox'],
});
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2, acceptDownloads: true });
await page.goto('https://carbon.now.sh/?' + params.toString(), { waitUntil: 'networkidle', timeout: 90000 });
const frame = page.locator('#export-container');
await frame.waitFor({ timeout: 60000 });
await page.waitForTimeout(2500); // fonts
// Carbon's own PNG export keeps the alpha channel; a page screenshot would
// pick up the page colour behind a transparent background.
await page.click('#export-menu');
const [download] = await Promise.all([
  page.waitForEvent('download', { timeout: 60000 }),
  page.click('#export-png'),
]);
await download.saveAs(process.argv[2] || 'promo.png');
await browser.close();
console.log('ok');
