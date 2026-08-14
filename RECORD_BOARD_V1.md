# Record Board V1 — evidence-driven geometry

This is the current SpeedPlaningLab design synthesis for the 160 km/h plane-towed record-board concept when high-fidelity CFD is not available in the short term.

## Geometry

- Overall length: **1550 mm**
- Maximum width: **295 mm**
- Tail/transom width: **275 mm**
- Reach full 295 mm width by approximately **280 mm from the tail**
- Hold near-full width through the central planing body; begin nose taper at approximately **950 mm from the tail**
- Nose tip width: approximately **50 mm**
- Bottom V: **0°** baseline
- Tail rocker: **0 mm**
- Aft bottom: deliberately **straight and flat for at least 500 mm**
- Nose rocker: **35 mm**, beginning at approximately **1100 mm from the tail**
- Thickness: approximately **18 mm** (structural placeholder)
- Deck crown: approximately **4 mm**
- Aft bottom/chine radius: **about 0.5–1.0 mm**; crisp release, not a fragile knife edge
- Rails become progressively more rounded forward of the active planing region
- No channels or transverse concave in V1
- No intentional longitudinal convexity or concavity in V1
- Rider/stance position: **adjustable**; do not manufacture around the legacy 250 mm pressure-balance point

## Design reference condition

Use **2.0° imposed running angle** as the experimental reference condition. This is not a claim that the ridden board will settle at 2°. The present rider/tow-handle pitching moment is not closed by the legacy equilibrium model.

For a 90 kg rider+board system at 160 km/h, using roughly 95% of weight as water-supported load, the published flat-plate reference for a 295 mm rectangular planing width is approximately:

- speed/width coefficient: about **26.1**
- wetted-length/width ratio: about **0.52**
- wetted length: about **150–155 mm**
- water drag: about **165–170 N**

These are reference-model values, not safety-certified predictions.

## Why 295 mm rather than 280 or 325 mm

Both the legacy model and the experiment-driven flat-plate correlation continue to favour narrower boards for drag. The design therefore does **not** treat 295 mm as a mathematical drag optimum.

The 295 mm choice is a compromise:

- 280 mm gives a higher beam-based speed coefficient and moves farther beyond the closest TN-2981 high-speed envelope.
- 300–325 mm moves closer to that speed envelope but, at 2° and reduced water-supported load, pushes the predicted wetted-length/width ratio toward or below the roughly 0.5 lower edge of the systematic flat-plate reference region.
- 295 mm stays close to both boundaries while preserving most of the narrow-board drag trend.

The tail is reduced to ~275 mm and widens gently to 295 mm because the expected active wetted patch is only around the first 150–200 mm. This keeps the actual working planform close to a rectangular plate while avoiding an unnecessarily locked-in full-width transom.

## Evidence hierarchy

1. **NACA TN-2981 — Weinstein & Kapryan (1953)**: high-speed rectangular flat-plate measurements of resistance, wetted length and pressure-center behaviour.
2. **NACA TN-3951 — Christopher (1957)**: flat-plate lift measurements up to 170 ft/s (~187 km/h), spanning the 160 km/h target speed.
3. **NACA TN-509 — Shoemaker (1934)**: flat and 10/20/30° deadrise planing surfaces with resistance, wetted length and center-of-pressure measurements.
4. **NACA TN-3939 / TR-1355 — Shuford**: flat/V planing theory checked against experiment; sharp versus slightly rounded chine effects.
5. **NASA Memo 1-25-59L — Mottard (1959)**: longitudinal convexity increased wetted length, decreased lift/drag, moved pressure center forward, and affected low-trim behaviour.
6. **Sottorf flat-plate work / NACA TM-1061**: independent pressure, lift, resistance and center-of-pressure evidence.

## What is intentionally not optimized yet

- physical stance / rider combined CG
- tow-handle height and pitch moment
- dynamic pitch stability / porpoising
- roll/yaw stability and fin requirement
- structural impact loads
- rider aerodynamic posture
- shallow V as a control/ride-quality trade
- mild longitudinal concavity as a moment/drag trade

## Current decision

Build/design around **1550 × 295 mm**, **275 mm tail**, **flat 0° V**, **straight aft bottom**, **hard aft chines**, and **adjustable rider position**. Use SpeedPlaningLab's published-data layer for sensitivity work and treat the legacy fast equilibrium model as a secondary trend tool only at record speed.
