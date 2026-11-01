// End-to-end tests of real user operations through the UI, as different users.
//   node e2e/operations.mjs     (stack running with demo data: `make demo-enterprise`)
//
// 1. Two-person production change: an engineer files a production rollback, cannot decide
//    it themselves, production stays unchanged, an approver approves it in another browser
//    session, and production then serves the previous version.
// 2. Staging deploy and rollback from the UI, verified against the API.
// 3. A viewer sees no approval controls and a write attempt is refused with the role error.
import { chromium } from "playwright";

const WEB = process.env.EVALFORGE_WEB ?? "http://localhost:3000";
const API = process.env.EVALFORGE_API ?? "http://localhost:8000";
const PASS = process.env.EVALFORGE_E2E_PASSWORD ?? "evalforge-dev";
let failures = 0;
const ok = (m) => console.log(`ok   ${m}`);
const fail = (m) => { console.error(`FAIL ${m}`); failures++; };

async function apiAs(user) {
  const r = await fetch(`${API}/api/auth/login`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ username: user, password: PASS }),
  });
  const { token } = await r.json();
  return (path) => fetch(`${API}${path}`, { headers: { authorization: `Bearer ${token}` } }).then((x) => x.json());
}

async function session(browser, user) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`${WEB}/login`);
  await page.getByLabel("Username").fill(user);
  await page.getByLabel("Password").fill(PASS);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL(`${WEB}/`);
  return { page, errors, close: () => ctx.close() };
}

const browser = await chromium.launch();
const get = await apiAs("engineer");

// ---------------------------------------------------------------- 1. two-person production change
// A production rollback always needs an approver who did not file it.
const prodBefore = (await get("/api/deployments")).find(
  (d) => d.environment === "production" && d.role === "stable" && d.status === "active");
const prevDep = prodBefore?.previous_deployment_id
  && (await get("/api/deployments")).find((d) => d.id === prodBefore.previous_deployment_id);
if (!prodBefore || !prevDep) {
  fail("needs a production deployment with a previous version; run `make demo-enterprise` first");
} else {
  const eng = await session(browser, "engineer");
  await eng.page.goto(`${WEB}/deployments`);
  await eng.page.getByRole("button", { name: "Roll back", exact: true }).last().click();
  await eng.page.getByRole("status").filter({ hasText: "Request filed" }).waitFor({ timeout: 15000 })
    .then(() => ok("engineer's production rollback became an approval request"),
          () => fail("no approval request notice"));
  const still = (await get("/api/deployments")).find(
    (d) => d.environment === "production" && d.role === "stable" && d.status === "active");
  still?.id === prodBefore.id ? ok("production unchanged until approved") : fail("production changed before approval");
  await eng.page.goto(`${WEB}/approvals`);
  (await eng.page.getByText("You filed this request, so another approver must decide it.").count()) > 0
    ? ok("requester sees separation-of-duties notice") : fail("requester notice missing");
  if (await eng.page.getByRole("button", { name: "Approve" }).count()) fail("engineer was offered Approve");

  const appr = await session(browser, "approver");
  await appr.page.goto(`${WEB}/approvals`);
  const item = appr.page.locator("li", { hasText: "Rollback" })
    .filter({ has: appr.page.getByRole("button", { name: "Approve" }) }).first();
  await item.getByLabel("Decision reason").fill("e2e: approved rollback");
  await item.getByRole("button", { name: "Approve" }).click();
  await appr.page.locator("li", { hasText: "e2e: approved rollback" }).getByText("executed")
    .waitFor({ timeout: 60000 })
    .then(() => ok("approver approved in a separate session; history shows executed"),
          () => fail("approval did not reach executed"));
  const after = (await get("/api/deployments")).filter(
    (d) => d.environment === "production" && d.role === "stable" && d.status === "active");
  after.length === 1 && after[0].version === prevDep.version
    ? ok(`production rolled back v${prodBefore.version} -> v${prevDep.version}`)
    : fail(`production is ${after.map((d) => "v" + d.version)}, expected v${prevDep.version}`);
  if (eng.errors.length + appr.errors.length) fail(`client errors: ${[...eng.errors, ...appr.errors]}`);
  await eng.close(); await appr.close();
}

// ---------------------------------------------------------------- 2. staging deploy + rollback
{
  const eng = await session(browser, "engineer");
  const before = (await get("/api/deployments")).find((d) => d.environment === "staging" && d.role === "stable" && d.status === "active");
  const name = before?.model_name;
  const versions = (await get("/api/registry/models")).find((m) => m.name === name)?.versions ?? [];
  const candidate = versions.find((v) => v.version !== before?.version);
  if (!candidate) {
    fail("no second version to deploy to staging");
  } else {
    await eng.page.goto(`${WEB}/registry`);
    await eng.page.getByRole("button", { name: new RegExp(`^v${candidate.version}\\b`) }).first().click();
    await eng.page.getByRole("button", { name: "Deploy to staging" }).click();
    await eng.page.getByRole("status").waitFor({ timeout: 30000 });
    let now = (await get("/api/deployments")).find((d) => d.environment === "staging" && d.role === "stable" && d.status === "active");
    now?.version === candidate.version ? ok(`staging deploy of v${candidate.version} from the registry`)
      : fail(`staging is v${now?.version}, expected v${candidate.version}`);
    await eng.page.goto(`${WEB}/deployments`);
    await eng.page.getByRole("button", { name: "Roll back", exact: true }).first().click();
    await eng.page.waitForTimeout(3000);
    now = (await get("/api/deployments")).find((d) => d.environment === "staging" && d.role === "stable" && d.status === "active");
    before && now?.version === before.version ? ok(`rollback restored staging v${before.version}`)
      : fail(`after rollback staging is v${now?.version}, expected v${before?.version}`);
    const actives = (await get("/api/deployments")).filter((d) => d.environment === "staging" && d.role === "stable" && d.status === "active");
    actives.length === 1 ? ok("exactly one active stable staging deployment")
      : fail(`${actives.length} active stable staging deployments`);
  }
  if (eng.errors.length) fail(`client errors: ${eng.errors}`);
  await eng.close();
}

// ---------------------------------------------------------------- 3. viewer is read-only
{
  const v = await session(browser, "viewer");
  await v.page.goto(`${WEB}/approvals`);
  await v.page.getByText("only approvers can decide").waitFor()
    .then(() => ok("viewer told only approvers can decide"), () => fail("viewer notice missing"));
  if (await v.page.getByRole("button", { name: "Approve" }).count()) fail("viewer was offered Approve");
  await v.page.goto(`${WEB}/registry`);
  await v.page.getByRole("button", { name: /^v\d+\b/ }).first().click();
  await v.page.getByRole("button", { name: "Deploy to staging" }).click();
  await v.page.getByRole("alert").filter({ hasText: "Requires role 'engineer'" }).waitFor({ timeout: 10000 })
    .then(() => ok("viewer write attempt refused with the role error"), () => fail("viewer write not refused"));
  if (v.errors.length) fail(`client errors: ${v.errors}`);
  await v.close();
}

await browser.close();
console.log(failures ? `OPERATIONS E2E FAILED (${failures})` : "OPERATIONS E2E PASSED");
process.exit(failures ? 1 : 0);
