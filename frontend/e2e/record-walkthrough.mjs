// Records the README walkthrough: drives the running app (API + web, demo data loaded) with real
// clicks and saves CDP screencast frames + frames.json. Usage: node e2e/record-walkthrough.mjs <out-dir>
import { chromium } from "playwright";
import { writeFileSync, mkdirSync } from "node:fs";
const OUT = process.argv[2]; mkdirSync(OUT, { recursive: true });
const W = "http://localhost:3000";
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
const p = await ctx.newPage();
// visible cursor + click ripple
await ctx.addInitScript(() => {
  addEventListener("DOMContentLoaded", () => {
    const c = document.createElement("div");
    c.style.cssText = "position:fixed;z-index:99999;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:rgba(255,255,255,.9);box-shadow:0 0 0 3px rgba(245,176,76,.55),0 2px 10px rgba(0,0,0,.5);pointer-events:none;transition:transform .12s;";
    const x = sessionStorage.getItem("cx"), y = sessionStorage.getItem("cy");
    if (x) { c.style.left = x + "px"; c.style.top = y + "px"; }
    document.body.appendChild(c);
    addEventListener("mousemove", (e) => { c.style.left = e.clientX + "px"; c.style.top = e.clientY + "px"; sessionStorage.setItem("cx", e.clientX); sessionStorage.setItem("cy", e.clientY); }, true);
    addEventListener("mousedown", () => { c.style.transform = "scale(.7)"; }, true);
    addEventListener("mouseup", () => { c.style.transform = "scale(1)"; }, true);
  });
});
const cdp = await ctx.newCDPSession(p);
const frames = []; let n = 0; let rec = false;
cdp.on("Page.screencastFrame", async (f) => {
  if (rec) { const name = `${String(n++).padStart(5, "0")}.jpg`; writeFileSync(`${OUT}/${name}`, Buffer.from(f.data, "base64")); frames.push([name, f.metadata.timestamp]); }
  await cdp.send("Page.screencastFrameAck", { sessionId: f.sessionId }).catch(() => {});
});
let mx = 960, my = 700;
async function move(loc, steps = 18) { const bb = await loc.boundingBox(); const x = bb.x + bb.width / 2, y = bb.y + bb.height / 2; await p.mouse.move(x, y, { steps }); mx = x; my = y; }
async function click(loc) { await move(loc); await p.waitForTimeout(80); await p.mouse.down(); await p.waitForTimeout(70); await p.mouse.up(); await loc.click({ trial: false, force: true }).catch(() => {}); }
const wait = (ms) => p.waitForTimeout(ms);

await p.goto(`${W}/login`, { waitUntil: "networkidle" });
await p.mouse.move(960, 760);
await cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
rec = true; const t0 = Date.now();
await wait(500);
const u = p.getByLabel(/username/i); await click(u); await u.pressSequentially("engineer", { delay: 35 });
const pw = p.getByLabel(/password/i); await click(pw); await pw.pressSequentially("evalforge-dev", { delay: 22 });
await click(p.getByRole("button", { name: /sign in/i })); await p.waitForURL(`${W}/`); await wait(1300);
// Copilot live investigation
await click(p.getByRole("link", { name: /MLOps Copilot/ }));
await p.getByLabel("Question").waitFor();
const q = p.getByLabel("Question"); await click(q); await q.fill("");
await q.pressSequentially("Why did my latest model perform worse?", { delay: 14 });
await click(p.getByRole("button", { name: "Investigate", exact: true }));
await p.getByText(/Investigation steps/).waitFor({ timeout: 30000 });
await p.waitForFunction(() => !document.body.innerText.includes("Working…"), null, { timeout: 30000 });
await wait(1500);
await p.mouse.wheel(0, 500); await wait(900);
// Pipeline studio: React Flow DAG + lineage
await click(p.getByRole("link", { name: /Pipeline studio/ })); await p.locator(".react-flow__node").first().waitFor(); await wait(700);
await click(p.locator(".react-flow__node").nth(4)); await wait(1100);
// Drift monitoring: PSI history heatmap
await click(p.getByRole("link", { name: /Drift monitoring/ })); await p.locator("canvas").first().waitFor(); await wait(800);
await p.mouse.move(1200, 700, { steps: 10 });
for (let i = 0; i < 8; i++) { await p.mouse.wheel(0, 140); await wait(60); }
await wait(600); await p.mouse.move(1150, 640, { steps: 14 }); await wait(900);
// Evaluation heatmap
await click(p.getByRole("link", { name: /Evaluation workbench/ })); await p.locator("tbody tr").first().waitFor(); await wait(500);
await click(p.locator("tbody tr").nth(1)); await wait(900);
for (let i = 0; i < 5; i++) { await p.mouse.wheel(0, 120); await wait(60); }
await p.mouse.move(1000, 700, { steps: 14 }); await wait(1300);
rec = false; await cdp.send("Page.stopScreencast");
writeFileSync(`${OUT}/frames.json`, JSON.stringify(frames));
console.log("frames", frames.length, "seconds", (Date.now() - t0) / 1000);
await b.close();
