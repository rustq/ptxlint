// Render promo.txt through carbon.now.sh and screenshot the export frame.
import { chromium } from 'playwright-core';
import { readFileSync } from 'fs';

const code = readFileSync(new URL('./promo.txt', import.meta.url), 'utf8').trimEnd();
const params = new URLSearchParams({
  bg: 'rgba(125,90,255,1)',
  t: 'dracula-pro',
  wt: 'none',
  l: 'application/x-sh',
  width: '680',
  ds: 'true',
  dsyoff: '20px',
  dsblur: '68px',
  wc: 'true',
  wa: 'true',
  pv: '48px',
  ph: '56px',
  ln: 'false',
  fm: 'JetBrains Mono',
  fs: '14px',
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
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
await page.goto('https://carbon.now.sh/?' + params.toString(), { waitUntil: 'networkidle', timeout: 90000 });
const frame = page.locator('#export-container');
await frame.waitFor({ timeout: 60000 });
await page.waitForTimeout(2500); // fonts
await frame.screenshot({ path: process.argv[2] || 'promo.png', omitBackground: false });
await browser.close();
console.log('ok');
