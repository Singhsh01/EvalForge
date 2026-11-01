#!/usr/bin/env node
/**
 * Turns brand/raw/<id>.(png|jpg|webp) into web-ready AVIF + WebP at 960/1600/2560 px in
 * public/brand/, plus public/brand/manifest.json (sizes, dominant colour, tiny blur placeholder).
 * Run after `brand.mjs generate`. Requires the `sharp` devDependency; not used at runtime.
 */
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, parse } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const raw = join(root, "brand/raw");
const out = join(root, "public/brand");
const WIDTHS = [960, 1600, 2560];
if (!existsSync(raw)) { console.log("brand/raw/ is empty — nothing to optimize (the app uses its CSS background)."); process.exit(0); }
mkdirSync(out, { recursive: true });
const manifest = { generated_at: new Date().toISOString(), assets: {} };
for (const f of readdirSync(raw).filter((f) => /\.(png|jpe?g|webp)$/i.test(f))) {
  const id = parse(f).name;
  const img = sharp(join(raw, f)).rotate();
  const meta = await img.metadata();
  const files = [];
  for (const w of WIDTHS.filter((w) => w <= (meta.width ?? 0) || w === WIDTHS[0])) {
    for (const [fmt, opts] of [["avif", { quality: 50, effort: 6 }], ["webp", { quality: 72 }]]) {
      const name = `${id}-${w}.${fmt}`;
      const info = await img.clone().resize({ width: w, withoutEnlargement: true })[fmt](opts).toFile(join(out, name));
      files.push({ src: `/brand/${name}`, width: info.width, format: fmt, bytes: info.size });
    }
  }
  const { dominant } = await img.stats();
  const blur = await img.clone().resize(24).webp({ quality: 40 }).toBuffer();
  manifest.assets[id] = {
    width: meta.width, height: meta.height,
    dominant: `rgb(${dominant.r} ${dominant.g} ${dominant.b})`,
    placeholder: `data:image/webp;base64,${blur.toString("base64")}`,
    files,
  };
  console.log(`${id}: ${files.map((x) => `${x.width}w.${x.format} ${(x.bytes / 1024).toFixed(0)}KB`).join(", ")}`);
}
writeFileSync(join(out, "manifest.json"), JSON.stringify(manifest, null, 2));
console.log("wrote public/brand/manifest.json");
