# Motion fintech landing page — build prompts

## What we're combining
- **Motion reference (video):** cards enter from scattered off-screen positions with 3D depth (rotateX/Y, translateZ), staggered timing, ease-out-expo snap into a bento grid. This assembly moment is the signature — not decoration, it IS the design.
- **Color reference (screenshot):** warm cream background, charcoal text, three soft pastel accents used only in data viz, mint-green for positive deltas, black pill CTAs.

## Design tokens (use these exact values)

```
Background:      #F6F2EC (warm cream)
Surface/card:     #FDFCFA
Text primary:     #17181C
Text secondary:   #6B6A66
Accent lavender:  #A99EF2
Accent coral:     #F2916B
Accent pink:      #F0A8C0
Positive/mint:    #4CAF7D
CTA (solid):      #111111 bg / #FFFFFF text, full pill radius
Border/hairline:  rgba(23,24,28,0.08)
Display font:     a confident geometric sans (Space Grotesk / General Sans)
Body font:        Inter
```

Card shadow should be barely-there: `0 1px 2px rgba(23,24,28,0.04), 0 8px 24px rgba(23,24,28,0.06)` — no heavy glassmorphism, no dark glow.

---

## PROMPT 1 — Full build (for Claude / Cursor / v0)

```
Build a motion-driven fintech landing page (single page, React + Tailwind + 
Framer Motion). Subject: [YOUR PRODUCT NAME] — an MSME credit-evaluation 
platform that scores small businesses using GST, UPI, Account Aggregator, 
and EPFO data instead of traditional paperwork.

COLOR SYSTEM (use exactly):
- Background: #F6F2EC (warm cream, not white)
- Card surface: #FDFCFA
- Text primary: #17181C, text secondary: #6B6A66
- Three data-viz accents only, never used for UI chrome: lavender #A99EF2, 
  coral #F2916B, pink #F0A8C0
- Positive-delta green: #4CAF7D
- CTA buttons: solid #111111 background, white text, full pill radius
- Shadows are barely visible: 0 1px 2px rgba(23,24,28,0.04), 0 8px 24px 
  rgba(23,24,28,0.06). No dark mode, no glassmorphism, no neon glow.

HERO SECTION — this is the signature moment, spend the most effort here:
- On page load, render 5-6 floating dashboard cards (a mini bar chart card, 
  a credit-score radial card, a transaction-list card, a GST-status card, 
  a stat card) scattered at different starting positions, opacity 0, 
  rotated in 3D (rotateX between -20 and 20deg, rotateY between -25 and 
  25deg), translated on the Z axis so they start "further away" and smaller.
- Animate each card into its final bento-grid position using Framer Motion, 
  staggered by 80-120ms per card, duration 0.9-1.2s, ease [0.16, 1, 0.3, 1] 
  (ease-out-expo feel — fast start, soft decelerating snap, no bounce).
- Headline text ("Credit scoring for businesses banks can't see yet" or 
  similar) fades/slides up 20px, starting slightly before the cards, so 
  copy reads first and the visual assembles around it.
- Give the whole hero container perspective: 1200px (CSS) so the 3D 
  rotations read correctly.
- Cards should have a very subtle continuous float (translateY sine loop, 
  amplitude ~4px, different phase per card) once settled, so the page 
  doesn't feel static.

BELOW THE FOLD:
- A "how it works" section: 3-4 steps, each revealed on scroll (Framer 
  Motion whileInView, fade + translateY 24px, once: true), NOT numbered 
  01/02/03 unless the steps are genuinely sequential.
- A data-sources strip (GST / UPI / Account Aggregator / EPFO) as small 
  pill badges with the lavender/coral/pink accents rotating per badge.
- A stats row (3 metrics) using tabular mono numerals for the figures.
- Footer CTA: repeat the black pill button, cream background.

INTERACTION DETAILS:
- Buttons: scale 0.97 on tap, no color change beyond a slight lighten on 
  hover.
- Use Lenis or Framer's built-in smooth scroll if available; otherwise 
  native scroll is fine — don't hijack scroll entirely.
- Respect prefers-reduced-motion: fall back to a simple fade, no 3D 
  transforms, if the user has it set.

Do not use a dark background. Do not use a single orange/terracotta 
accent as the only color. Do not default to numbered 01/02/03 step 
markers unless steps are truly sequential. Build as a single-file React 
component with Tailwind classes and Framer Motion for all animation.
```

---

## PROMPT 2 — Just the hero card-assembly animation (if you want to prototype the signature moment first, in isolation)

```
Build ONLY a hero section: React + Framer Motion, Tailwind for layout.
Cream background #F6F2EC. 6 floating cards (rectangles with rounded-2xl, 
white #FDFCFA fill, faint border, containing placeholder fintech UI: a 
mini chart, a score ring, a transaction row list, a GST status chip) that 
start off their final grid position — scattered, small opacity, rotated 
in 3D, pushed back in Z — and animate into a clean bento grid on mount.
Stagger 100ms apart, ease [0.16, 1, 0.3, 1], duration ~1s. Add perspective 
to the container. After landing, each card gently floats on a sine loop. 
Headline centered above/behind the grid, cream text-primary #17181C, 
large geometric sans, fades up slightly ahead of the cards. No dark mode.
```

---

## PROMPT 3 — Copy direction (if you want help with headline/subhead wording)

```
Write hero copy for an MSME credit-scoring platform. Audience: bank credit 
officers and NBFC underwriters evaluating small businesses that lack 
traditional financial paperwork. The product uses GST filings, UPI 
transaction history, Account Aggregator bank data, and EPFO payroll data 
instead. Tone: confident, plain-spoken, not hype-y — this is a compliance/
risk-adjacent audience, avoid startup marketing clichés like "revolutionize" 
or "unlock". One headline (under 10 words), one subhead (under 20 words), 
one CTA button label (2-3 words).
```

---

## Notes for whichever tool you paste this into
- If using **v0**: paste Prompt 1 as-is, it handles Framer Motion + Tailwind natively.
- If using **Cursor**: paste Prompt 1, then follow up with Prompt 2 alone in a fresh chat if the hero animation needs isolated iteration — easier to debug the 3D timing without the rest of the page in the way.
- If using **Claude**: you can paste Prompt 1 directly here and I'll build it as a live artifact.