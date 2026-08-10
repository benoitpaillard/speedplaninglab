# Speed Planing Lab v4 — Streamlit

A deployable customer-facing planing-hydrodynamics explorer for the aircraft-towed high-speed wakeboard project.

## What works now

- **OpenPlaning 0.4.8** as the primary analytical engine.
- Internal Stage-0 screening model as an explicitly labelled fallback.
- Live controls for speed, mass, beam, LCG/VCG, deadrise, tow geometry, rider CdA, roughness, and fluid properties.
- Equilibrium trim, wetted length/area, LCP, water resistance, aero drag, tow force, Fn_B, lambda and CL.
- Visible Savitsky/OpenPlaning range checks — no silent extrapolation.
- Porpoising/stability indicators where the engine provides them.
- Running-geometry schematic.
- Speed sweeps with force and attitude plots.
- Lightweight beam × LCG design search with ranked tow-force results.
- A/B design snapshots.
- JSON design import/export and CSV sweep export.
- CFD-surrogate plug-in slot that appears automatically when `speedlab/cfd_surrogate.py` is added.

## Local run

```bash
./run_local.sh
```

Then open `http://localhost:8501`.

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Fastest public deployment: Streamlit Community Cloud

1. Put this folder in a GitHub repository.
2. Open Streamlit Community Cloud and create an app.
3. Select the repository and branch.
4. Entrypoint: `streamlit_app.py`.
5. Choose Python **3.12** in Advanced settings.
6. Deploy.

The repository already contains the required `requirements.txt` and `.streamlit/config.toml`.

## Docker

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## CFD integration

When the CFD workstream produces a surrogate, copy it to:

```text
speedlab/cfd_surrogate.py
```

and expose:

```python
VERSION = "surrogate-v1"

def evaluate(design_dict: dict) -> dict:
    ...
```

The Streamlit UI detects the file automatically and enables the CFD model in its engine selector. See `speedlab/cfd_surrogate.example.py` and `CFD_INTEGRATION.md`.

## Important model limitation

The 160 km/h wakeboard case lies far outside important conventional Savitsky correlation coordinates, especially trim and beam Froude number. The app deliberately displays those violations. Analytical output should therefore be used for design-space exploration and CFD prioritisation, not as a final safety or structural basis.
