# NACA TN-2981 experimental source notes

Source: Irving Weinstein and Walter J. Kapryan, **The High-speed Planing Characteristics of a Rectangular Flat Plate over a Wide Range of Trim and Wetted Length**, NACA TN-2981, 1953. NASA NTRS document 19930083703.

Official record: https://ntrs.nasa.gov/citations/19930083703

## Current status

The original report is a primary experimental source for high-speed rectangular flat-plate planing. It remains a key validation source for SpeedPlaningLab.

The scan/OCR of Table I does **not** preserve all row/trim grouping cleanly enough to justify treating the extracted rows as a verified 2-degree dataset. The earlier `table_i_trim_2deg.csv` file has therefore been retired.

`table_i_ocr_provisional.csv` preserves the useful numerical transcription for provenance and later manual verification, but it is explicitly **not used by the live dashboard or optimizer**. Its trim grouping must be checked against the original page image before any row is promoted to verified experimental data.

## Live experiment-driven reference

The live dashboard now uses `published_flatplate_reference.py` instead of the provisional OCR table. That module keeps the evidence hierarchy explicit:

1. TN-2981: primary high-speed flat-plate experiments and the closest classical experimental envelope to the record-board problem.
2. Shuford NACA flat-plate theory/correlation: developed and checked against broad flat-plate planing experiments.
3. P. Ward Brown, Davidson Laboratory SIT-DL-71-1463: systematic controlled planing-surface experiments and the adopted Shuford-Brown lift, drag, and Schoenherr-friction formulation.
4. Christopher, NACA TN-3951: independent flat-plate lift measurements at speeds up to 170 ft/s (about 187 km/h), spanning the record-board target speed.

The live reference solves wetted length from the published lift relation for a selected running angle, then evaluates pressure/induced drag plus Schoenherr skin friction. The dashboard displays the 2–10 degree experiment-backed range and clearly flags any prediction below 2 degrees as extrapolation.

## Center of pressure

TN-2981 reports an approximately constant center-of-pressure / mean-wetted-length relationship near 0.71 through the lower-trim range. This is retained as published context, but it is **not silently substituted for unreadable Table I cells** and is not used to repair the current rider/tow equilibrium model.

## Measurement uncertainty

TN-2981 states approximate measurement accuracies of ±0.15 lb load, ±0.15 lb resistance, ±0.50 ft-lb trimming moment, ±0.25 in wetted length, ±0.05 in draft, ±0.10 degree trim, and ±0.20 ft/s speed.

TN-2981 also cautions that mean-wetted-length / beam values below 0.5 have marginal wetted-area measurement accuracy.

## Data-integrity policy

No ambiguous OCR cell is promoted to verified experimental data by inference. Raw/provisional transcriptions stay separated from live model data. Derived values must be reproducible from documented equations and every live correlation must remain traceable to a published source.