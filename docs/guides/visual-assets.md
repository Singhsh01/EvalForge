# Optional visual assets with Higgsfield

EvalForge can use AI-generated backgrounds and illustrations for visual branding. This is
**entirely optional**:

- Higgsfield is not a runtime, build or CI dependency. No code path imports it.
- With no generated assets, `public/brand/manifest.json` is `{ "assets": {} }` and the app
  renders its CSS ambient background, exactly as before.
- Generated images are only decoration. Charts, workflow diagrams and animations are real UI
  built with Apache ECharts, React Flow and Motion, all driven by API data.

## Rules

1. **No generation without approval.** `brand.mjs generate` refuses to run unless you pass
   `--approved-credits N`.
2. **Cost is checked before every generation.** `generate` re-runs `higgsfield generate cost`
   immediately before `generate create` and aborts if the estimate is above N.
3. `brand.mjs cost` only estimates. It never creates a job.

## One-time setup (on your machine)

```bash
npm i -g @higgsfield/cli          # official CLI (binary: higgsfield / hf)
higgsfield auth login             # opens a browser for OAuth
higgsfield workspace list         # then select one; the CLI requires a workspace:
higgsfield workspace set <workspace_id>
npx skills add higgsfield-ai/skills   # official companion skills for coding agents
higgsfield account status         # shows your credit balance
```

Check the model ids and parameters used in `frontend/brand/prompts.json` before estimating:

```bash
higgsfield model list --image
higgsfield model get nano_banana_2
```

## Workflow

```bash
cd frontend
npm run brand:list                       # the 3 defined assets and where they are used
npm run brand:cost                       # credit estimate per asset + total (no jobs)
npm run brand:generate -- hero-background --approved-credits <N>   # only after you approve N
npm run brand:optimize                   # AVIF + WebP at 960/1600/2560 px + manifest.json
```

`brand:generate` saves raw output to `frontend/brand/raw/` (git-ignored). `brand:optimize`
writes the web assets to `frontend/public/brand/` together with a manifest that records sizes,
the dominant colour and a 24-px blur placeholder.

## How the frontend uses them

`components/brand-background.tsx`:

- fetches `/brand/manifest.json`. If there is no `hero-background` entry it renders nothing;
- shows the dominant colour and blur placeholder first, then a responsive `<picture>`
  (AVIF, then WebP) loaded at low fetch priority, fading in with Motion. There is no motion
  when the user prefers reduced motion;
- always draws a dark gradient scrim over the image so that white type keeps strong contrast;
- can be switched off with `NEXT_PUBLIC_EVALFORGE_BRAND_ASSETS=off`.

`observability-illustration` and `copilot-illustration` are defined for empty states but are not
wired in yet. Add them once you have generated and approved them.

## Prompt guidelines

The prompts ask for dark navy, low-detail, text-free and logo-free images with large calm
areas, so content stays readable on top. Edit `brand/prompts.json` to change them. Re-run
`brand:cost` after any change, because cost depends on the model and its parameters.
