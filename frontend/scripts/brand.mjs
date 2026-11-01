#!/usr/bin/env node
/**
 * Optional Higgsfield asset workflow for EvalForge. NOT used at build or run time.
 *
 *   node scripts/brand.mjs list                              show the asset definitions
 *   node scripts/brand.mjs cost [id...]                      estimate credits (creates no job)
 *   node scripts/brand.mjs generate <id> --approved-credits N
 *        re-checks the cost first and refuses if it exceeds N; downloads to brand/raw/<id>.*
 *
 * Generation never happens without an explicit --approved-credits value chosen by a human
 * after reading the `cost` output.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const spec = JSON.parse(readFileSync(join(root, "brand/prompts.json"), "utf8"));
const [cmd = "help", ...rest] = process.argv.slice(2);

function cli(args) {
  try {
    return execFileSync("higgsfield", [...args, "--json", "--no-color"], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  } catch (e) {
    if (e.code === "ENOENT") die("Higgsfield CLI not found. Install it with: npm i -g @higgsfield/cli  (optional — EvalForge works without it)");
    die(`higgsfield ${args[0]} ${args[1] ?? ""} failed:\n${e.stderr || e.message}`);
  }
}
function die(msg) { console.error(msg); process.exit(1); }
function asset(id) { return spec.assets.find((a) => a.id === id) ?? die(`unknown asset "${id}" — see: node scripts/brand.mjs list`); }
function argsFor(a) {
  const model = a.model ?? spec.defaults.model;
  const params = { ...spec.defaults.params, ...(a.params ?? {}), prompt: a.prompt };
  return [model, ...Object.entries(params).flatMap(([k, v]) => [`--${k}`, String(v)])];
}
/** Find the credit figure in the CLI's JSON without assuming one exact schema. */
function credits(json) {
  const seen = [];
  const walk = (o, path = "") => {
    if (o && typeof o === "object") for (const [k, v] of Object.entries(o)) walk(v, `${path}.${k}`);
    else if (typeof o === "number" && /credit|cost|price|amount/i.test(path)) seen.push(o);
  };
  let parsed; try { parsed = JSON.parse(json); } catch { die(`could not parse cost output:\n${json}`); }
  walk(parsed);
  if (!seen.length) die(`no credit figure found in cost output — inspect it manually:\n${json}`);
  return Math.max(...seen);
}
function cost(a) { return credits(cli(["generate", "cost", ...argsFor(a)])); }

if (cmd === "list") {
  for (const a of spec.assets) console.log(`${a.id.padEnd(28)} ${a.model ?? spec.defaults.model}  ${a.use}`);
} else if (cmd === "cost") {
  const ids = rest.length ? rest : spec.assets.map((a) => a.id);
  let total = 0;
  for (const id of ids) { const c = cost(asset(id)); total += c; console.log(`${id.padEnd(28)} ${c} credits`); }
  console.log(`${"total".padEnd(28)} ${total} credits\n\nNothing was generated. To generate one asset after approving its cost:\n  node scripts/brand.mjs generate <id> --approved-credits <N>`);
} else if (cmd === "generate") {
  const id = rest.find((r) => !r.startsWith("--"));
  const i = rest.indexOf("--approved-credits");
  const approved = i >= 0 ? Number(rest[i + 1]) : NaN;
  if (!id) die("usage: node scripts/brand.mjs generate <id> --approved-credits N");
  if (!Number.isFinite(approved)) die("Refusing to generate: pass --approved-credits N after reviewing `node scripts/brand.mjs cost`.");
  const a = asset(id);
  const c = cost(a);
  console.log(`${id}: current estimate ${c} credits (approved ≤ ${approved})`);
  if (c > approved) die("Refusing to generate: the estimate exceeds the approved credits.");
  const out = cli(["generate", "create", ...argsFor(a), "--wait"]);
  mkdirSync(join(root, "brand/raw"), { recursive: true });
  writeFileSync(join(root, `brand/raw/${id}.job.json`), out);
  const url = (out.match(/https:\/\/[^\s"']+\.(?:png|jpe?g|webp)/i) ?? [])[0];
  if (!url) die(`job finished but no image URL found; raw output saved to brand/raw/${id}.job.json`);
  const res = await fetch(url);
  if (!res.ok) die(`download failed: ${res.status}`);
  const ext = url.split("?")[0].split(".").pop();
  writeFileSync(join(root, `brand/raw/${id}.${ext}`), Buffer.from(await res.arrayBuffer()));
  console.log(`saved brand/raw/${id}.${ext} — now run: npm run brand:optimize`);
} else {
  console.log(readFileSync(fileURLToPath(import.meta.url), "utf8").split("*/")[0].replace(/^#!.*\n\/\*\*?/, "").replace(/^ \* ?/gm, ""));
}
