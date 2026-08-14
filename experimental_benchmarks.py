import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

G = 9.81

# Coverage ranges transcribed from the published planing-hydrodynamics literature.
# These are applicability envelopes, not digitized point-by-point measurements yet.
BENCHMARKS = [
    {
        "name": "Weinstein & Kapryan (1953)",
        "report": "NACA TN-2981",
        "bottom_v": [0.0],
        "trim_min": 2.0,
        "trim_max": 30.0,
        "fnb_min": 4.15,
        "fnb_max": 25.50,
        "quantities": "lift, resistance, wetted length, center of pressure",
        "source": "https://ntrs.nasa.gov/citations/19930083703",
        "role": "Primary high-speed flat-plate benchmark",
    },
    {
        "name": "Shoemaker (1934)",
        "report": "NACA TN-509",
        "bottom_v": [0.0, 10.0, 20.0, 30.0],
        "trim_min": 2.0,
        "trim_max": 12.0,
        "fnb_min": 1.65,
        "fnb_max": 7.33,
        "quantities": "resistance, wetted length, center of pressure",
        "source": "https://ntrs.nasa.gov/citations/19930081288",
        "role": "Flat/V-bottom cross-check",
    },
    {
        "name": "Sottorf (1932/1944)",
        "report": "NACA TM-1061",
        "bottom_v": [0.0],
        "trim_min": 1.2,
        "trim_max": 11.3,
        "fnb_min": 2.33,
        "fnb_max": 5.54,
        "quantities": "pressure, lift, resistance, center of pressure",
        "source": "https://ntrs.nasa.gov/citations/20030065218",
        "role": "Independent flat-plate cross-check",
    },
    {
        "name": "Sambraus (1938)",
        "report": "NACA TM-848",
        "bottom_v": [0.0],
        "trim_min": 2.3,
        "trim_max": 19.9,
        "fnb_min": 3.50,
        "fnb_max": 13.19,
        "quantities": "high-Froude flat-plate planing characteristics",
        "source": "https://ntrs.nasa.gov/",
        "role": "High-speed flat-plate cross-check",
    },
    {
        "name": "Kapryan & Boyd (1955)",
        "report": "NACA TN-3477",
        "bottom_v": [0.0, 20.0, 40.0],
        "trim_min": 4.0,
        "trim_max": 30.0,
        "fnb_min": 6.83,
        "fnb_max": 15.28,
        "quantities": "hydrodynamic pressure distributions",
        "source": "https://ntrs.nasa.gov/citations/19930084267",
        "role": "Pressure-distribution CFD validation",
    },
    {
        "name": "Shuford (1957)",
        "report": "NACA TN-3939",
        "bottom_v": [0.0, 20.0, 40.0],
        "trim_min": 8.0,
        "trim_max": 34.0,
        "fnb_min": 9.07,
        "fnb_max": 18.65,
        "quantities": "lift and center of pressure; planform/cross-section effects",
        "source": "https://ntrs.nasa.gov/citations/19930084991",
        "role": "Extended theory/experiment cross-check",
    },
]


def beam_froude(speed_kmh, width_m):
    v = float(speed_kmh) / 3.6
    return v / math.sqrt(G * max(float(width_m), 1e-9))


def width_for_fnb(speed_kmh, fnb):
    v = float(speed_kmh) / 3.6
    return v * v / (G * float(fnb) * float(fnb))


def _distance_to_interval(value, lo, hi):
    if lo <= value <= hi:
        return 0.0
    if value < lo:
        return (lo - value) / max(hi - lo, 1e-9)
    return (value - hi) / max(hi - lo, 1e-9)


def _v_distance(value, tested_values):
    return min(abs(float(value) - float(v)) for v in tested_values)


def _coverage_score(benchmark, fnb, trim, bottom_v):
    fnb_d = _distance_to_interval(fnb, benchmark["fnb_min"], benchmark["fnb_max"])
    trim_d = _distance_to_interval(trim, benchmark["trim_min"], benchmark["trim_max"])
    v_d = _v_distance(bottom_v, benchmark["bottom_v"]) / 10.0
    # Transparent heuristic used only for ranking benchmark relevance.
    score = 100.0 / (1.0 + 4.0 * fnb_d + 3.0 * trim_d + 2.0 * v_d)
    return max(0.0, min(100.0, score))


def render_experimental_benchmarks(*, speed_kmh, width_m, bottom_v_deg, running_angle_deg):
    st.divider()
    st.subheader("Published experimental benchmarks")
    st.caption(
        "This section compares the current board with published towing-tank test envelopes. "
        "For now it uses documented test ranges; point-by-point digitized measurements can be added next."
    )

    fnb = beam_froude(speed_kmh, width_m)
    ranked = []
    for b in BENCHMARKS:
        ranked.append((
            _coverage_score(b, fnb, running_angle_deg, bottom_v_deg),
            b,
        ))
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_score, best = ranked[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current speed / width number", f"{fnb:.2f}")
    c2.metric("Current board angle", f"{running_angle_deg:.2f}°")
    c3.metric("Current bottom V", f"{bottom_v_deg:.1f}°")
    c4.metric("Closest published benchmark", best["report"])

    if best["report"] == "NACA TN-2981":
        max_fnb = best["fnb_max"]
        target_width = width_for_fnb(speed_kmh, max_fnb)
        over = 100.0 * (fnb / max_fnb - 1.0)
        if fnb <= max_fnb:
            st.success(
                f"At this speed and width, the speed/width number is inside the Weinstein–Kapryan flat-plate test range (up to {max_fnb:.2f})."
            )
        else:
            st.warning(
                f"The current speed/width number is {over:.1f}% above the Weinstein–Kapryan flat-plate maximum of {max_fnb:.2f}. "
                f"At {speed_kmh:.0f} km/h, a width of about {target_width * 1000:.0f} mm would sit on that published high-speed limit."
            )
        if running_angle_deg < best["trim_min"]:
            st.warning(
                f"The predicted running angle ({running_angle_deg:.2f}°) is below the TN-2981 measured trim range starting near {best['trim_min']:.1f}°. "
                "This low-angle extrapolation is currently a larger concern than total board length."
            )

    rows = []
    for score, b in ranked:
        rows.append({
            "Benchmark": f"{b['name']} — {b['report']}",
            "Bottom V tested": ", ".join(f"{v:g}°" for v in b["bottom_v"]),
            "Board angle range": f"{b['trim_min']:g}–{b['trim_max']:g}°",
            "Speed/width range": f"{b['fnb_min']:.2f}–{b['fnb_max']:.2f}",
            "Measured": b["quantities"],
            "Relevance score": round(score),
            "Use": b["role"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    fig = go.Figure()
    for i, (_, b) in enumerate(ranked):
        label = b["report"]
        fig.add_trace(go.Scatter(
            x=[b["fnb_min"], b["fnb_max"]],
            y=[label, label],
            mode="lines+markers",
            line=dict(width=8),
            marker=dict(size=8),
            name=label,
            hovertemplate=(
                f"{b['name']}<br>Speed/width range: {b['fnb_min']:.2f}–{b['fnb_max']:.2f}<br>"
                f"Board angle: {b['trim_min']:g}–{b['trim_max']:g}°<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.add_vline(x=fnb, line_width=3, line_dash="dash", annotation_text="current board")
    fig.update_layout(
        height=390,
        xaxis_title="Speed / width planing number",
        yaxis_title="Published experiment",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Design guidance from the published coverage")
    st.write(
        f"**Primary benchmark:** {best['name']} ({best['report']}) — {best['role']}."
    )
    st.write(
        "For the current flat-board concept, the most valuable next step is to digitize its measured lift, resistance, wetted-length and center-of-pressure curves and compare the dashboard model against those points directly."
    )

    with st.expander("Original report sources"):
        for _, b in ranked:
            st.markdown(f"- **{b['report']} — {b['name']}**: {b['source']}")
