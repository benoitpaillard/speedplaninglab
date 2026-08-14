import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq

G = 9.81


def shuford_brown_lift_coeff(wet_ratio, trim_deg, bottom_v_deg, speed_width_number):
    """Eq. (11) in Brown, Davidson Laboratory report SIT-DL-71-1463.

    Returns lift coefficient based on q*b^2 for an unflapped plane surface.
    The formulation combines Shuford dynamic lift with Brown's static term.
    """
    lam = float(wet_ratio)
    tau = math.radians(float(trim_deg))
    beta = math.radians(float(bottom_v_deg))
    cv = float(speed_width_number)
    if lam <= 0 or cv <= 0 or tau <= 0:
        return 0.0

    crossflow_cd = 1.33  # plane surface, any deadrise, Brown report p. 12
    bracket = (
        (1.0 - math.sin(beta)) * lam**2 / (1.0 + lam)
        + (crossflow_cd / math.pi) * lam * math.sin(tau) ** 2 * math.cos(beta)
        + 0.4 / math.cos(tau) * (lam / cv) ** 2
    )
    return (math.pi / 4.0) * math.sin(2.0 * tau) * math.cos(tau) * bracket


def _schoenherr_cf(reynolds):
    """Brown Eq. (13): 0.242/sqrt(Cf) = log10(Re*Cf)."""
    re = max(float(reynolds), 1.0e4)

    def residual(cf):
        return 0.242 / math.sqrt(cf) - math.log10(re * cf)

    return brentq(residual, 1.0e-5, 0.03)


def reference_at_trim(
    *,
    target_lift_coeff,
    trim_deg,
    bottom_v_deg,
    speed_width_number,
    speed_mps,
    width_m,
    water_density,
    water_kinematic_viscosity,
):
    """Solve the published flat-plate correlation for wetted length and drag."""
    target = float(target_lift_coeff)
    if target <= 0:
        return None

    def residual(lam):
        return shuford_brown_lift_coeff(
            lam, trim_deg, bottom_v_deg, speed_width_number
        ) - target

    lo, hi = 0.03, 0.5
    while residual(hi) < 0 and hi < 20.0:
        hi *= 1.6
    if residual(lo) * residual(hi) > 0:
        return None

    lam = float(brentq(residual, lo, hi))
    tau = math.radians(float(trim_deg))
    beta = math.radians(float(bottom_v_deg))
    wetted_length = lam * float(width_m)
    re = max(float(speed_mps) * wetted_length / float(water_kinematic_viscosity), 1.0e4)
    cf = _schoenherr_cf(re)

    # Brown Eq. (12), no flap: pressure/induced drag + Schoenherr skin friction.
    drag_coeff = (
        target * math.tan(tau)
        + cf * lam / (math.cos(tau) * max(math.cos(beta), 1.0e-6))
    )
    q = 0.5 * float(water_density) * float(speed_mps) ** 2
    drag_n = drag_coeff * q * float(width_m) ** 2

    return {
        "trim_deg": float(trim_deg),
        "wet_ratio": lam,
        "wetted_length_m": wetted_length,
        "drag_coeff": drag_coeff,
        "water_drag_n": drag_n,
        "skin_friction_coeff": cf,
        "reynolds": re,
    }


def render_published_flatplate_reference(
    *,
    speed_kmh,
    width_m,
    bottom_v_deg,
    running_angle_deg,
    current_lift_coeff,
    current_water_drag_n,
    current_wetted_length_m,
    water_density,
    water_kinematic_viscosity,
):
    st.markdown("#### Experiment-driven flat-plate reference")
    st.caption(
        "This is the continuous published-data reference used for design comparison. "
        "It uses the Shuford–Brown lift relation adopted by Davidson Laboratory, Brown's "
        "drag relation and Schoenherr skin friction. It is kept separate from the dashboard's "
        "current fast model."
    )

    speed_mps = float(speed_kmh) / 3.6
    cv = speed_mps / math.sqrt(G * max(float(width_m), 1.0e-9))

    # Brown's systematic experiments used 2, 4, 6, 8 and 10 degrees. Add a denser
    # curve between them for readability, but clearly flag sub-2-degree extrapolation.
    sweep_angles = np.linspace(2.0, 10.0, 65)
    rows = []
    for angle in sweep_angles:
        ref = reference_at_trim(
            target_lift_coeff=current_lift_coeff,
            trim_deg=float(angle),
            bottom_v_deg=bottom_v_deg,
            speed_width_number=cv,
            speed_mps=speed_mps,
            width_m=width_m,
            water_density=water_density,
            water_kinematic_viscosity=water_kinematic_viscosity,
        )
        if ref is not None:
            rows.append(ref)

    if not rows:
        st.warning("The published flat-plate reference could not be solved for this setup.")
        return

    ref_df = pd.DataFrame(rows)
    at_two = reference_at_trim(
        target_lift_coeff=current_lift_coeff,
        trim_deg=2.0,
        bottom_v_deg=bottom_v_deg,
        speed_width_number=cv,
        speed_mps=speed_mps,
        width_m=width_m,
        water_density=water_density,
        water_kinematic_viscosity=water_kinematic_viscosity,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current speed / width number", f"{cv:.2f}")
    c2.metric("Published-data floor", "2.0°")
    if at_two:
        c3.metric("2° water contact", f"{at_two['wetted_length_m'] * 1000:.0f} mm")
        c4.metric("2° water drag", f"{at_two['water_drag_n']:.0f} N")

    if running_angle_deg < 2.0:
        st.warning(
            f"The dashboard predicts a {running_angle_deg:.2f}° running angle, below the "
            "2° lower edge of the high-speed flat-plate data used to validate this reference. "
            "The 2° result is therefore the nearest experiment-backed design point, not proof "
            "that the board will run below 2°."
        )

    p1, p2 = st.columns(2)
    with p1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ref_df["trim_deg"],
                y=1000.0 * ref_df["wetted_length_m"],
                mode="lines",
                name="Published flat-plate reference",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[running_angle_deg],
                y=[1000.0 * current_wetted_length_m],
                mode="markers",
                marker=dict(size=14, symbol="star"),
                name="Current dashboard model",
            )
        )
        fig.add_vline(x=2.0, line_dash="dash", annotation_text="measured-data floor")
        fig.update_layout(
            height=390,
            xaxis_title="Board running angle (deg)",
            yaxis_title="Board length touching water (mm)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ref_df["trim_deg"],
                y=ref_df["water_drag_n"],
                mode="lines",
                name="Published flat-plate reference",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[running_angle_deg],
                y=[current_water_drag_n],
                mode="markers",
                marker=dict(size=14, symbol="star"),
                name="Current dashboard model",
            )
        )
        fig.add_vline(x=2.0, line_dash="dash", annotation_text="measured-data floor")
        fig.update_layout(
            height=390,
            xaxis_title="Board running angle (deg)",
            yaxis_title="Water drag (N)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Evidence used")
    evidence = pd.DataFrame(
        [
            {
                "Evidence": "Weinstein & Kapryan — NACA TN-2981",
                "What it contributes": "High-speed flat-plate planing measurements and the closest published envelope to the record board",
                "Status": "Primary experiment",
            },
            {
                "Evidence": "Shuford — NACA TN-3939 / TN-3233",
                "What it contributes": "Flat-plate lift theory checked against experiment over broad planing conditions",
                "Status": "Experiment-validated correlation",
            },
            {
                "Evidence": "Brown — Davidson SIT-DL-71-1463",
                "What it contributes": "Systematic controlled tests; adopted Shuford lift relation, fitted static term, drag relation and Schoenherr friction",
                "Status": "Primary experiment + fitted correlation",
            },
            {
                "Evidence": "Christopher — NACA TN-3951",
                "What it contributes": "Independent high-speed flat-plate lift tests up to 170 ft/s (~187 km/h)",
                "Status": "Primary high-speed check",
            },
        ]
    )
    st.dataframe(evidence, hide_index=True, use_container_width=True)

    st.caption(
        "Important: this reference is strongest for lift, wetted length and broad drag trends. "
        "It does not fix the missing rider/tow-handle pitching-moment physics in the current "
        "equilibrium solver, and it should not be treated as a safety-certified prediction."
    )
