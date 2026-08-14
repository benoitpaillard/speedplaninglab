from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

G = 9.81
DATA_PATH = Path(__file__).resolve().parent / "data" / "experiments" / "tn2981" / "table_i_trim_2deg.csv"


def load_trim2_data():
    return pd.read_csv(DATA_PATH)


def _current_dimensionless(*, speed_kmh, width_m, water_density, current_clb,
                           current_water_drag_n, current_wetted_length_m):
    v = float(speed_kmh) / 3.6
    b = float(width_m)
    q = 0.5 * float(water_density) * v * v
    cv = v / math.sqrt(G * b)
    clb = float(current_clb)
    cdb = float(current_water_drag_n) / max(q * b * b, 1e-12)
    lam = float(current_wetted_length_m) / max(b, 1e-12)
    ca = 0.5 * clb * cv * cv
    return dict(CA=ca, Cv=cv, CLb=clb, CDb=cdb, lm_over_b=lam)


def _nearest_row(df, current):
    ca_span = max(df["CA"].max() - df["CA"].min(), 1e-9)
    cv_span = max(df["Cv"].max() - df["Cv"].min(), 1e-9)
    distance = np.sqrt(
        ((df["CA"] - current["CA"]) / ca_span) ** 2
        + ((df["Cv"] - current["Cv"]) / cv_span) ** 2
    )
    return df.loc[distance.idxmin()]


def _reference_extrapolation(df, current_clb):
    # Use the nearest-load series and the three highest-speed / lowest-CL points.
    # This is deliberately a local linear extrapolation, not a new empirical law.
    unique_loads = np.sort(df["CA"].unique())
    nearest_load = float(unique_loads[np.argmin(np.abs(unique_loads - current_clb * 0 + 4.26))])
    group = df[np.isclose(df["CA"], nearest_load)].sort_values("CLb")
    if len(group) < 3:
        return None
    local = group.head(3)
    lo, hi = float(local["CLb"].min()), float(local["CLb"].max())
    # Refuse long extrapolations: this is intended only for our present point just
    # beyond the fastest TN-2981 2-degree measurement.
    if current_clb < 0.70 * lo or current_clb > 1.30 * hi:
        return None
    p_wet = np.polyfit(local["CLb"], local["lm_over_b"], 1)
    p_drag = np.polyfit(local["CLb"], local["CDb"], 1)
    return {
        "CA": nearest_load,
        "lm_over_b": float(np.polyval(p_wet, current_clb)),
        "CDb": float(np.polyval(p_drag, current_clb)),
        "min_CLb": lo,
        "max_CLb": hi,
    }


def render_tn2981_digitized(*, speed_kmh, width_m, running_angle_deg,
                             current_clb, current_water_drag_n,
                             current_wetted_length_m, water_density):
    df = load_trim2_data()
    cur = _current_dimensionless(
        speed_kmh=speed_kmh,
        width_m=width_m,
        water_density=water_density,
        current_clb=current_clb,
        current_water_drag_n=current_water_drag_n,
        current_wetted_length_m=current_wetted_length_m,
    )
    nearest = _nearest_row(df, cur)

    st.markdown("#### Digitized TN-2981 measurements — 2° flat plate")
    st.caption(
        "These are transcribed Table I measurements, not traced pixels. The plotted "
        "coefficients are recomputed from the report definitions. The current dashboard "
        "point is shown for comparison, but its predicted running angle is not forced to 2°."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current speed / width number", f"{cur['Cv']:.2f}")
    c2.metric("Current load level", f"{cur['CA']:.2f}")
    c3.metric("Current water-contact / width", f"{cur['lm_over_b']:.2f}")
    c4.metric("Nearest measured load series", f"{nearest['CA']:.2f}")

    if running_angle_deg < 2.0:
        st.warning(
            f"The dashboard currently predicts {running_angle_deg:.2f}° running angle. "
            "The digitized measurements below are at 2.00°, so they are a nearby experimental "
            "reference, not a direct replacement for the equilibrium calculation."
        )

    nearest_load = float(df.iloc[(df["CA"] - cur["CA"]).abs().argsort()[:1]]["CA"].iloc[0])
    near_load_df = df[np.isclose(df["CA"], nearest_load)].sort_values("CLb")

    p1, p2 = st.columns(2)
    with p1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["CLb"], y=df["lm_over_b"], mode="markers",
            marker=dict(size=8, color=df["CA"], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Load level")),
            text=df["Cv"],
            hovertemplate="Measured 2°<br>lift coeff %{x:.4f}<br>water contact / width %{y:.2f}<br>speed/width %{text:.2f}<extra></extra>",
            name="TN-2981 measurements",
        ))
        fig.add_trace(go.Scatter(
            x=near_load_df["CLb"], y=near_load_df["lm_over_b"], mode="lines+markers",
            line=dict(width=3), marker=dict(size=9),
            name=f"Closest load series ({nearest_load:.2f})",
        ))
        fig.add_trace(go.Scatter(
            x=[cur["CLb"]], y=[cur["lm_over_b"]], mode="markers",
            marker=dict(size=15, symbol="star"), name="Current dashboard prediction",
            hovertemplate="Current model<br>lift coeff %{x:.4f}<br>water contact / width %{y:.2f}<extra></extra>",
        ))
        fig.update_layout(height=410, xaxis_title="Lift coefficient", yaxis_title="Board length touching water / board width")
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["CLb"], y=df["CDb"], mode="markers",
            marker=dict(size=8, color=df["CA"], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Load level")),
            text=df["Cv"],
            hovertemplate="Measured 2°<br>lift coeff %{x:.4f}<br>water-drag coeff %{y:.4f}<br>speed/width %{text:.2f}<extra></extra>",
            name="TN-2981 measurements",
        ))
        fig.add_trace(go.Scatter(
            x=near_load_df["CLb"], y=near_load_df["CDb"], mode="lines+markers",
            line=dict(width=3), marker=dict(size=9),
            name=f"Closest load series ({nearest_load:.2f})",
        ))
        fig.add_trace(go.Scatter(
            x=[cur["CLb"]], y=[cur["CDb"]], mode="markers",
            marker=dict(size=15, symbol="star"), name="Current dashboard prediction",
            hovertemplate="Current model<br>lift coeff %{x:.4f}<br>water-drag coeff %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(height=410, xaxis_title="Lift coefficient", yaxis_title="Water-drag coefficient")
        st.plotly_chart(fig, use_container_width=True)

    # Give a carefully bounded 2-degree reference at the present lift coefficient.
    # For the present wakeboard, the nearest experimental load series is CA=4.26.
    loads = np.sort(df["CA"].unique())
    ref_load = float(loads[np.argmin(np.abs(loads - cur["CA"]))])
    ref_group = df[np.isclose(df["CA"], ref_load)].sort_values("CLb")
    reference = None
    if len(ref_group) >= 3:
        local = ref_group.head(3)
        min_cl = float(local["CLb"].min())
        if 0.70 * min_cl <= cur["CLb"] <= 1.30 * float(local["CLb"].max()):
            wet_fit = np.polyfit(local["CLb"], local["lm_over_b"], 1)
            drag_fit = np.polyfit(local["CLb"], local["CDb"], 1)
            reference = (
                float(np.polyval(wet_fit, cur["CLb"])),
                float(np.polyval(drag_fit, cur["CLb"])),
            )

    if reference is not None:
        wet_ratio, drag_coeff = reference
        v = float(speed_kmh) / 3.6
        q = 0.5 * float(water_density) * v * v
        ref_wet_m = wet_ratio * float(width_m)
        ref_drag_n = drag_coeff * q * float(width_m) ** 2
        st.info(
            f"**2° experimental reference near the current load:** a short local extrapolation "
            f"of the three fastest points in the CA={ref_load:.2f} series gives roughly "
            f"{ref_wet_m * 1000:.0f} mm water-contact length and {ref_drag_n:.0f} N water drag "
            f"at the current lift coefficient. This is a 2° reference only — it does not prove "
            f"that the board will actually settle at 2°."
        )

    with st.expander("Digitized rows"):
        shown = df.copy()
        shown["CA"] = shown["CA"].round(2)
        shown["Cv"] = shown["Cv"].round(2)
        shown["CLb"] = shown["CLb"].round(5)
        shown["CDb"] = shown["CDb"].round(5)
        shown["lm_over_b"] = shown["lm_over_b"].round(2)
        st.dataframe(
            shown[["trim_flag", "CA", "Cv", "CR", "lm_over_b", "CLb", "CDb", "confidence", "note"]],
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "TN-2981 warns that water-contact/width ratios below 0.5 have marginal wetted-area measurement accuracy."
        )
