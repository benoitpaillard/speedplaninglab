# NACA TN-2981 digitized data

Source: Irving Weinstein and Walter J. Kapryan, **The High-speed Planing Characteristics of a Rectangular Flat Plate over a Wide Range of Trim and Wetted Length**, NACA TN-2981, 1953. NASA NTRS document 19930083703.

Official record: https://ntrs.nasa.gov/citations/19930083703

## What is digitized

`table_i_trim_2deg.csv` is the first design-relevant slice transcribed from **Table I, experimental planing data**, for the 2 degree trim condition. This is intentionally the first slice because the record wakeboard currently predicts a running angle below 2 degrees, making 2 degrees the nearest directly measured trim in TN-2981.

The primary transcribed quantities are:

- `CA`: beam loading / load coefficient used by the report
- `Cv`: speed coefficient (beam-based Froude number)
- `CR`: resistance coefficient used by the report
- `lm_over_b`: mean wetted length divided by beam

The following columns are **recomputed from the report definitions**, rather than re-OCRed, to reduce transcription error:

- `CLb = 2 * CA / Cv^2`
- `CDb = 2 * CR / Cv^2`
- `CLS = CLb / (lm/b)`
- `CDS = CDb / (lm/b)`

Rows marked `2a` reproduce the report's footnote-a test condition (alternate average kinematic viscosity). They remain part of the 2 degree trim dataset and are explicitly flagged.

## Why center of pressure and draft are not in this first CSV

The indexed/OCR text for the center-of-pressure and draft columns is substantially less reliable than the load/speed/resistance/wetted-length columns. Rather than silently guess values, they are being added only when they can be verified confidently against the original table/figures. The report states that the center-of-pressure to wetted-length ratio is approximately 0.71 through 9 degrees trim, but this relationship is not substituted for measured table values here.

## Measurement uncertainty stated in TN-2981

The report states approximate measurement accuracies of ±0.15 lb load, ±0.15 lb resistance, ±0.50 ft-lb trimming moment, ±0.25 in wetted length, ±0.05 in draft, ±0.10 degree trim, and ±0.20 ft/s speed.

TN-2981 also cautions that mean-wetted-length/beam values below 0.5 have marginal wetted-area measurement accuracy. SpeedPlaningLab should surface this warning when using such rows.

## Digitization policy

The original NASA report is a US Government work and the NTRS record marks it public-use permitted. Every digitized row must remain traceable to the report. Ambiguous OCR is left blank or omitted rather than inferred. Derived values must be reproducible from documented formulas.
