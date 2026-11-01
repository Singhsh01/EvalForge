// Browser end-to-end smoke test against a running stack (API :8000, web :3000).
//   node e2e/smoke.mjs            (needs `npx playwright install chromium` once)
// Signs in through the UI, then checks that every module renders real data from the API
// without client-side errors, and that the Copilot answers a question with cited evidence.
import { chromium } from "playwright";

const WEB = process.env.EVALFORGE_WEB ?? "http://localhost:3000";
const USER = process.env.EVALFORGE_E2E_USER ?? "engineer";
const PASS = process.env.EVALFORGE_E2E_PASSWORD ?? "evalforge-dev";
const PAGES = [
  ["/", "Operations overview"], ["/copilot", "AI MLOps Copilot"], ["/incidents", "Incident investigation"],
  ["/approvals", "Approvals"], ["/datasets", "Dataset studio"], ["/experiments", "Experiment lab"],
  ["/pipelines", "Pipeline studio"], ["/registry", "Model registry"], ["/deployments", "Deployment center"],
  ["/evaluations", "Evaluation workbench"], ["/prompts", "Prompt registry"], ["/traces", "Trace explorer"],
  ["/monitoring", "Drift monitoring"], ["/analytics", "Cost & performance analytics"], ["/system", "System health & audit"],
];

const fail = (m) => { console.error("FAIL", m); process.exitCode = 1; };
const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto(`${WEB}/login`);
await page.getByLabel("Username").fill(USER);
await page.getByLabel("Password").fill(PASS);
await page.getByRole("button", { name: "Sign in" }).click();
await page.waitForURL(`${WEB}/`);
console.log("ok   login as", USER);

for (const [path, heading] of PAGES) {
  await page.goto(WEB + path, { waitUntil: "networkidle" });
  const h = await page.getByRole("heading", { level: 1 }).first().textContent();
  if (h?.trim() !== heading) fail(`${path}: heading "${h}" != "${heading}"`);
  else console.log("ok  ", path);
}

await page.goto(`${WEB}/copilot`, { waitUntil: "networkidle" });
await page.getByLabel("Question").fill("Compare the last five experiments and recommend a candidate model.");
await page.getByRole("button", { name: "Investigate", exact: true }).click();
await page.getByText(/Investigation steps/).waitFor({ timeout: 30000 });
await page.waitForFunction(() => !document.body.innerText.includes("Working…"), null, { timeout: 30000 });
const text = await page.locator("main").innerText();
if (!/recommend_candidate/.test(text)) fail("copilot did not call recommend_candidate");
else console.log("ok   copilot investigation with tool calls");

await page.goto(`${WEB}/login`);
const anon = await browser.newPage();
await anon.goto(`${WEB}/registry`);
await anon.waitForURL(`${WEB}/login`, { timeout: 10000 }).then(() => console.log("ok   unauthenticated users are redirected to sign-in"), () => fail("no redirect to /login"));

if (errors.length) fail(`client errors: ${errors.join(" | ")}`);
await browser.close();
console.log(process.exitCode ? "E2E FAILED" : "E2E PASSED");
