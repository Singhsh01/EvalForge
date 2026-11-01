"use client";

import { motion, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";

type Asset = { width: number; height: number; dominant: string; placeholder: string; files: { src: string; width: number; format: string }[] };

/**
 * Optional Higgsfield-generated background. Loads /brand/manifest.json if it exists; otherwise
 * renders nothing and the CSS `.ambient` gradient remains the background. Set
 * NEXT_PUBLIC_EVALFORGE_BRAND_ASSETS=off to disable it entirely.
 */
export function BrandBackground({ id = "hero-background" }: { id?: string }) {
  const [asset, setAsset] = useState<Asset | null>(null);
  const [loaded, setLoaded] = useState(false);
  const reduce = useReducedMotion();
  useEffect(() => {
    if (process.env.NEXT_PUBLIC_EVALFORGE_BRAND_ASSETS === "off") return;
    let cancelled = false;
    fetch("/brand/manifest.json", { cache: "force-cache" })
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => { if (!cancelled && m?.assets?.[id]) setAsset(m.assets[id]); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [id]);
  if (!asset) return null;
  const srcset = (fmt: string) => asset.files.filter((f) => f.format === fmt).map((f) => `${f.src} ${f.width}w`).join(", ");
  const fallback = asset.files.find((f) => f.format === "webp")?.src;
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 z-0 overflow-hidden"
      style={{ backgroundColor: asset.dominant, backgroundImage: `url(${asset.placeholder})`, backgroundSize: "cover", backgroundPosition: "center" }}>
      <motion.picture className="absolute inset-0 block"
        initial={{ opacity: 0, scale: reduce ? 1 : 1.04 }}
        animate={loaded ? { opacity: 1, scale: 1 } : {}}
        transition={{ duration: reduce ? 0 : 1.6, ease: [0.22, 1, 0.36, 1] }}>
        <source type="image/avif" srcSet={srcset("avif")} sizes="100vw" />
        <source type="image/webp" srcSet={srcset("webp")} sizes="100vw" />
        {/* eslint-disable-next-line @next/next/no-img-element -- pre-optimized responsive set */}
        <img src={fallback} alt="" decoding="async" fetchPriority="low" onLoad={() => setLoaded(true)}
          className="h-full w-full object-cover" />
      </motion.picture>
      {/* Readability scrim: keeps white type at AA contrast over any generated image. */}
      <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(7,10,22,0.78),rgba(7,10,22,0.88)_40%,rgba(7,10,22,0.94))]" />
    </div>
  );
}
