import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from published_flatplate_reference import reference_at_trim

G = 9.81

# Evidence-driven V1 geometry. These are design decisions, not outputs of the
# legacy pitch-equilibrium solver.
V1 = {
    "length_m": 1.550,
    "max_width_m": 0.295,
    "tail_width_m": 0.275,
    "full_width_x_m": 0.280,
    "nose_taper_x_m": 0.950,
    "nose_tip_width_m": 0.050,
    "nose_rocker_m": 0.035,
    "rocker_start_x_m": 1.100,
    "tail_rocker_m": 0.0,
    "thickness_m": 0.018,
    "deck_crown_m": 0.004,
    "bottom_v_deg": 0.0,
    "reference_trim_deg": 2.0,
    "aft_chine_radius_mm": "0.5–1.0",
    "aft_straight_bottom_m": 0.50,
}


def _ref(*, speed_kmh, width_m, trim_deg, lift_n, rho_w, nu_w, beta=0.0):
    v = float(speed_kmh) / 3.6
    cv = v / math.sqrt(G * float(width_m))
    q = 0.5 * float(rho_w) * v * v
    cl = float(lift_n) / max(q * float(width_m) ** 2, 1e-12)
    return reference_at_trim(
        target_lift_coeff=cl,
        trim_deg=float(trim_deg),
        bottom_v_deg=float(beta),
        speed_width_number=cv,
        speed_mps=v,
        width_m=float(width_m),
        water_density=float(rho_w),
        water_kinematic_viscosity=float(nu_w),
    )


def _width_study(speed_kmh, mass_kg, rho_w, nu_w, supported_fraction=0.95):
    lift_n = float(mass_kg) * G * float(supported_fraction)
    rows = []
    for width in np.arange(0.270, 0.321, 0.005):
        ref = _ref(
            speed_kmh=speed_kmh,
            width_m=width,
            trim_deg=2.0,
            lift_n=lift_n,
            rho_w=rho_w,
            nu_w=nu_w,
        )
        if ref is None:
            continue
        cv = (float(speed_kmh) / 3.6) / math.sqrt(G * width)
        rows.append(
            {
                "width_mm": 1000.0 * width,
                "Cv": cv,
                "wet_ratio": ref["wet_ratio"],
                "wet_mm": 1000.0 * ref["wetted_length_m"],
                "drag_n": ref["water_drag_n"],
                "near_TN2981_speed": cv <= 26.2,
                "wet_ratio_supported": ref["wet_ratio"] >= 0.50,
            }
        )
    return pd.DataFrame(rows)


def _trim_study(speed_kmh, mass_kg, rho_w, nu_w, width_m=V1["max_width_m"], supported_fraction=0.95):
    lift_n = float(mass_kg) * G * float(supported_fraction)
    rows = []
    for trim in np.arange(2.0, 4.01, 0.25):
        ref = _ref(
            speed_kmh=speed_kmh,
            width_m=width_m,
            trim_deg=trim,
            lift_n=lift_n,
            rho_w=rho_w,
            nu_w=nu_w,
        )
        if ref is not None:
            rows.append(
                {
                    "trim_deg": trim,
                    "wet_ratio": ref["wet_ratio"],
                    "wet_mm": 1000.0 * ref["wetted_length_m"],
                    "drag_n": ref["water_drag_n"],
                    "supported": ref["wet_ratio"] >= 0.50,
                }
            )
    return pd.DataFrame(rows)


def render_record_board_design(*, speed_kmh, mass_kg, rho_w, nu_w):
    st.subheader("Recommended record-board V1")
    st.caption(
        "This is the current design synthesis from published flat-plate experiments and the 1550 mm manufacturing limit. "
        "It deliberately does not use the legacy pitch-equilibrium optimum to choose the geometry."
    )

    width = V1["max_width_m"]
    tail = V1["tail_width_m"]
    speed_mps = float(speed_kmh) / 3.6
    cv = speed_mps / math.sqrt(G * width)
    nominal_lift = 0.95 * float(mass_kg) * G
    ref = _ref(
        speed_kmh=speed_kmh,
        width_m=width,
        trim_deg=V1["reference_trim_deg"],
        lift_n=nominal_lift,
        rho_w=rho_w,
        nu_w=nu_w,
    )

    a, b, c, d, e = st.columns(5)
    a.metric("Length", "1550 mm")
    b.metric("Maximum width", "295 mm")
    c.metric("Tail width", "275 mm")
    d.metric("Bottom", "flat · 0° V")
    e.metric("Reference condition", "2.0°")

    if ref is not None:
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Speed / width number", f"{cv:.2f}")
        f2.metric("Reference water contact", f"{1000 * ref['wetted_length_m']:.0f} mm")
        f3.metric("Water contact / width", f"{ref['wet_ratio']:.2f}")
        f4.metric("Reference water drag", f"{ref['water_drag_n']:.0f} N")

    st.info(
        "**Why 295 mm?** It is a compromise between the repeated narrow-is-fast trend and experimental support. "
        "At 160 km/h it stays close to the TN-2981 high-speed envelope, while at 2° the experiment-driven reference "
        "keeps wetted-length/width close to the ~0.5 lower boundary instead of relying on the much shorter patches produced at 3–4°."
    )

    geometry = pd.DataFrame(
        [
            ["Overall length", "1550 mm", "manufacturing limit; not a steady-drag variable"],
            ["Tail width", "275 mm", "slightly narrower for more progressive roll/edge control"],
            ["Maximum width", "295 mm", "reached by x ≈ 280 mm from tail"],
            ["Aft bottom", "straight + flat", "at least first 500 mm; no designed convexity/concavity"],
            ["Bottom V", "0°", "baseline with strongest flat-plate evidence; shallow V remains a future control experiment"],
            ["Aft chines", "hard, ~0.5–1 mm radius", "clean release without a fragile knife edge"],
            ["Forward rails", "progressively rounded", "reduce risk of an abrupt forward edge catch during disturbance"],
            ["Nose taper begins", "~950 mm from tail", "normally dry high-speed region"],
            ["Nose rocker", "35 mm", "starts ~1100 mm from tail"],
            ["Tail rocker", "0 mm", "keep active planing region geometrically straight"],
            ["Thickness", "~18 mm", "structural placeholder; hydrodynamically secondary in steady planing"],
            ["Stance / rider position", "adjustable", "do not lock it to the legacy 250 mm pressure-balance point"],
            ["Channels / transverse concave", "none", "insufficient evidence and adds ventilation/separation sensitivity"],
            ["Longitudinal concave", "0 mm for V1", "interesting later, but it changes pitch moment as well as drag"],
            ["Longitudinal convex", "0 mm for V1", "published tests show more wetted length and worse L/D than flat"],
        ],
        columns=["Feature", "V1 choice", "Reason"],
    )
    st.dataframe(geometry, hide_index=True, use_container_width=True)

    st.markdown("#### What the experiments say about width")
    width_df = _width_study(speed_kmh, mass_kg, rho_w, nu_w)
    if not width_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=width_df["width_mm"],
                y=width_df["drag_n"],
                mode="lines+markers",
                name="2° experiment-driven reference",
                customdata=np.stack([width_df["Cv"], width_df["wet_ratio"], width_df["wet_mm"]], axis=1),
                hovertemplate=(
                    "Width %{x:.0f} mm<br>Water drag %{y:.0f} N<br>"
                    "Speed/width %{customdata[0]:.2f}<br>Wetted length/width %{customdata[1]:.2f}<br>"
                    "Wetted length %{customdata[2]:.0f} mm<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=295, line_dash="dash", annotation_text="V1 = 295 mm")
        fig.update_layout(
            height=380,
            xaxis_title="Maximum/reference planing width (mm)",
            yaxis_title="Reference water drag at 2° (N)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The correlation still trends toward narrower boards. V1 does not pick 295 mm because it is the drag minimum; "
            "it picks it because it stays close to the high-speed test envelope and retains wetted-length/width ≈ 0.5 at the 2° reference condition."
        )

    st.markdown("#### Why 2° is a reference, not a claimed equilibrium")
    trim_df = _trim_study(speed_kmh, mass_kg, rho_w, nu_w)
    if not trim_df.empty:
        p1, p2 = st.columns(2)
        with p1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trim_df["trim_deg"], y=trim_df["wet_ratio"], mode="lines+markers"))
            fig.add_hline(y=0.50, line_dash="dash", annotation_text="~lower experimental support boundary")
            fig.update_layout(height=340, xaxis_title="Imposed running angle (deg)", yaxis_title="Wetted length / width")
            st.plotly_chart(fig, use_container_width=True)
        with p2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trim_df["trim_deg"], y=trim_df["drag_n"], mode="lines+markers"))
            fig.update_layout(height=340, xaxis_title="Imposed running angle (deg)", yaxis_title="Reference water drag (N)")
            st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "The rider + tow-line pitch moment is still unresolved. Published experiments tell us the water forces for an imposed attitude; "
        "they do not tell us where the ridden board will settle. V1 therefore uses adjustable stance/mounting and treats 2° as the closest useful experimental design condition."
    )

    with st.expander("Evidence hierarchy used for V1"):
        evidence = pd.DataFrame(
            [
                ["NACA TN-2981 · Weinstein & Kapryan", "rectangular flat plate", "high-speed resistance, wetted length, center of pressure", "primary design anchor"],
                ["NACA TN-3951 · Christopher", "flat plate to 170 ft/s (~187 km/h)", "high-speed lift speed-effect check", "supports use near 160 km/h"],
                ["NACA TN-509 · Shoemaker", "0–30° deadrise", "resistance, wetted length, center of pressure", "supports flat baseline"],
                ["NACA TN-3939 / TR-1355 · Shuford", "flat/V, sharp and slightly rounded chines", "lift, CP, chine/cross-section effects", "supports hard aft edges"],
                ["NASA Memo 1-25-59L · Mottard", "0° deadrise longitudinal convexity", "wetted length, drag, CP, trim", "supports zero convexity for V1"],
                ["Sottorf TM-1061 / related flat-plate tests", "flat and V surfaces", "pressure, lift, resistance, CP", "independent classical cross-check"],
            ],
            columns=["Source", "Geometry", "Measured/useful quantities", "Role in V1"],
        )
        st.dataframe(evidence, hide_index=True, use_container_width=True)
        st.markdown(
            "Primary source records: [TN-2981](https://ntrs.nasa.gov/citations/19930083703) · "
            "[TN-3951](https://ntrs.nasa.gov/citations/19930084670) · "
            "[TN-509](https://ntrs.nasa.gov/citations/19930081288) · "
            "[TR-1355](https://ntrs.nasa.gov/citations/19930092342) · "
            "[Mottard convexity study](https://ntrs.nasa.gov/citations/19980232090)"
        )

    st.error(
        "**Design status:** evidence-driven prototype geometry, not a safety-certified or ride-ready 160 km/h board. "
        "Dynamic pitch/porpoising, yaw/roll stability, rider aerodynamics, structural impact loads and tow-system loads remain separate validation problems."
    )
