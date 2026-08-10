import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq

st.set_page_config(page_title="Speed Planing Lab", page_icon="🌊", layout="wide")

try:
    import openplaning as op  # noqa: F401
    HAVE_OPENPLANING = True
except Exception:
    HAVE_OPENPLANING = False

G = 9.81


def savitsky_cl(lam, tau_deg, fn_b, beta_deg):
    cl0 = tau_deg**1.1 * (
        0.012 * lam**0.5 + 0.0055 * lam**2.5 / max(fn_b**2, 1e-9)
    )
    return cl0 - 0.0065 * beta_deg * max(cl0, 1e-12) ** 0.6


def screening(speed_kmh, mass, beam, lcg, beta, tow_angle, cda, rho_w, nu_w, rho_air, rough_um):
    v = speed_kmh / 3.6
    q = 0.5 * rho_w * v * v
    fn_b = v / math.sqrt(G * beam)
    weight = mass * G
    aero = 0.5 * rho_air * v * v * cda
    tow_rad = math.radians(tow_angle)

    def solve_lam(lift, tau):
        def f(lam):
            return savitsky_cl(lam, tau, fn_b, beta) * q * beam * beam - lift

        lo, hi = 1e-5, 1.0
        while f(hi) < 0 and hi < 64:
            hi *= 2
        if f(lo) * f(hi) > 0:
            raise RuntimeError("No wetted-length solution")
        return brentq(f, lo, hi)

    def state(tau):
        lift = weight
        for _ in range(30):
            lam = solve_lam(lift, tau)
            wet_len = lam * beam
            area = lam * beam * beam / max(math.cos(math.radians(beta)), 1e-6)
            re = max(v * wet_len / nu_w, 1e4)
            cf = 0.075 / (math.log10(re) - 2.0) ** 2
            krel = max(rough_um * 1e-6 / max(wet_len, 1e-6), 0.0)
            dcf = min(0.0015, 0.044 * krel ** (1 / 3)) if krel > 0 else 0.0
            friction = q * (cf + dcf) * area
            pressure = lift * math.tan(math.radians(tau))
            water = friction + pressure
            horizontal = water + aero
            new_lift = max(weight - horizontal * math.tan(tow_rad), 0.45 * weight)
            if abs(new_lift - lift) < 1e-6 * weight:
                lift = new_lift
                break
            lift = 0.5 * (lift + new_lift)

        lcp = lam * beam * (
            0.75 - 1.0 / (5.21 * (fn_b / max(lam, 1e-9)) ** 2 + 2.39)
        )
        return lam, wet_len, area, lcp, friction, pressure, water, horizontal, lift

    def moment_residual(tau):
        return state(tau)[3] - lcg

    taus = np.linspace(0.12, 8.0, 80)
    vals = [moment_residual(float(t)) for t in taus]
    tau = None
    for a, b, fa, fb in zip(taus[:-1], taus[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
            tau = brentq(moment_residual, float(a), float(b))
            break
    if tau is None:
        raise RuntimeError("No trim equilibrium in 0.12–8°")

    lam, wet_len, area, lcp, friction, pressure, water, horizontal, lift = state(tau)
    cl = lift / (q * beam * beam)
    tension = horizontal / max(math.cos(tow_rad), 1e-6)
    checks = {
        "0.6 ≤ FnB ≤ 13": 0.6 <= fn_b <= 13,
        "2° ≤ trim ≤ 15°": 2 <= tau <= 15,
        "λ ≤ 4": lam <= 4,
        "0.0338 ≤ CL ≤ 0.18": 0.0338 <= cl <= 0.18,
        "0° ≤ deadrise ≤ 20°": 0 <= beta <= 20,
    }
    score = int(round(100 * sum(checks.values()) / len(checks)))
    return {
        "trim_deg": tau,
        "water_resistance_n": water,
        "aero_drag_n": aero,
        "tow_force_n": tension,
        "wetted_length_m": wet_len,
        "wetted_area_m2": area,
        "lcp_m": lcp,
        "fn_b": fn_b,
        "lambda": lam,
        "cl": cl,
        "friction_n": friction,
        "pressure_n": pressure,
        "lift_n": lift,
        "weight_n": weight,
        "checks": checks,
        "validity_score": score,
    }


def evaluate(**p):
    return screening(**p)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def board_half_width(
    x,
    length,
    beam,
    tail_width_ratio,
    max_beam_pos,
    taper_start,
    tip_width_ratio,
):
    """Parametric directional planform."""
    u = np.asarray(x, dtype=float) / max(length, 1e-9)
    max_beam_pos = max(min(max_beam_pos, taper_start - 0.03), 0.05)
    taper_start = min(max(taper_start, max_beam_pos + 0.03), 0.97)

    aft_t = _smoothstep(u / max(max_beam_pos, 1e-6))
    aft = tail_width_ratio + (1.0 - tail_width_ratio) * aft_t

    nose_t = _smoothstep((u - taper_start) / max(1.0 - taper_start, 1e-6))
    nose = 1.0 - (1.0 - tip_width_ratio) * nose_t

    width_factor = np.where(
        u <= max_beam_pos,
        aft,
        np.where(u <= taper_start, 1.0, nose),
    )
    return 0.5 * beam * np.maximum(width_factor, 0.03)


def _camera_for(view_name):
    views = {
        "Perspective": dict(
            eye=dict(x=1.45, y=-1.55, z=0.82),
            center=dict(x=0.02, y=0.0, z=-0.04),
        ),
        "Top": dict(
            eye=dict(x=0.02, y=-0.02, z=2.55),
            center=dict(x=0.02, y=0.0, z=0.0),
        ),
        "Side": dict(
            eye=dict(x=0.05, y=-2.55, z=0.28),
            center=dict(x=0.02, y=0.0, z=-0.02),
        ),
        "Chase": dict(
            eye=dict(x=-1.75, y=-1.05, z=0.48),
            center=dict(x=0.12, y=0.0, z=-0.03),
        ),
    }
    return views.get(view_name, views["Perspective"])


def make_board_3d(
    *,
    length,
    beam,
    trim_deg,
    beta_deg,
    wet_len,
    lcg,
    lcp,
    tow_angle_deg,
    tail_width_ratio,
    max_beam_pos,
    taper_start,
    tip_width_ratio,
    nose_rocker,
    rocker_start,
    tail_rocker,
    thickness,
    deck_crown_height,
    view_name="Perspective",
    show_forces=True,
    lift_n=None,
    weight_n=None,
    water_drag_n=None,
    aero_drag_n=None,
    tow_force_n=None,
):
    """True-scale parametric board geometry with analytical running-state overlays."""
    nx, ny = 96, 39
    xs = np.linspace(0.0, length, nx)
    yn = np.linspace(-1.0, 1.0, ny)
    X = np.repeat(xs[:, None], ny, axis=1)

    half = board_half_width(
        xs, length, beam, tail_width_ratio, max_beam_pos, taper_start, tip_width_ratio
    )
    Y = half[:, None] * yn[None, :]

    tau = math.radians(trim_deg)
    beta = math.radians(beta_deg)
    wet = min(max(wet_len, 0.0), length)

    def rocker_phys(xv):
        uv = np.asarray(xv, dtype=float) / max(length, 1e-9)
        nose_t = np.clip((uv - rocker_start) / max(1.0 - rocker_start, 1e-6), 0.0, 1.0)
        nose = nose_rocker * nose_t**2.2
        tail_t = np.clip((0.24 - uv) / 0.24, 0.0, 1.0)
        tail = tail_rocker * tail_t**2.0
        return nose + tail

    # Running-attitude convention: the predicted forward wet limit touches z=0
    # on the centerline. The hydrodynamic model does not currently solve heave.
    heave = -(wet * math.tan(tau) + float(rocker_phys(wet)))

    Zbottom = heave + X * math.tan(tau) + rocker_phys(X) + np.abs(Y) * math.tan(beta)
    crown_shape = np.clip(
        1.0 - (Y / np.maximum(half[:, None], 1e-6)) ** 2, 0.0, 1.0
    )
    Ztop = Zbottom + thickness + deck_crown_height * crown_shape

    def half_width_at(xv):
        return float(
            board_half_width(
                np.array([xv]),
                length,
                beam,
                tail_width_ratio,
                max_beam_pos,
                taper_start,
                tip_width_ratio,
            )[0]
        )

    def bottom_z(xv, yv=0.0):
        return (
            heave
            + xv * math.tan(tau)
            + float(rocker_phys(xv))
            + abs(yv) * math.tan(beta)
        )

    def deck_z(xv, yv=0.0):
        hw = max(half_width_at(xv), 1e-6)
        crown = deck_crown_height * max(1.0 - (yv / hw) ** 2, 0.0)
        return bottom_z(xv, yv) + thickness + crown

    fig = go.Figure()

    # Free-surface plane only: no schematic spray/wake.
    wx = np.linspace(-0.18 * length, 1.08 * length, 2)
    wy = np.linspace(-1.08 * beam, 1.08 * beam, 2)
    WX, WY = np.meshgrid(wx, wy)
    fig.add_trace(
        go.Surface(
            x=WX,
            y=WY,
            z=np.zeros_like(WX),
            surfacecolor=np.zeros_like(WX),
            colorscale=[[0, "#178ebd"], [1, "#178ebd"]],
            showscale=False,
            opacity=0.20,
            hoverinfo="skip",
            name="Water",
        )
    )

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Zbottom,
            surfacecolor=X,
            colorscale=[[0, "#0b141d"], [0.55, "#162733"], [1, "#2b4352"]],
            showscale=False,
            opacity=0.995,
            lighting=dict(
                ambient=0.40, diffuse=0.80, roughness=0.42, specular=0.50, fresnel=0.12
            ),
            lightposition=dict(x=-2, y=-3, z=7),
            hovertemplate="Bottom<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>",
            name="Bottom",
        )
    )
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Ztop,
            surfacecolor=X,
            colorscale=[
                [0, "#0f3846"],
                [0.50, "#176071"],
                [0.82, "#248494"],
                [1, "#164b5b"],
            ],
            showscale=False,
            opacity=1.0,
            lighting=dict(
                ambient=0.48, diffuse=0.84, roughness=0.28, specular=0.72, fresnel=0.16
            ),
            lightposition=dict(x=-2, y=-4, z=8),
            hovertemplate="Deck<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>",
            name="Deck",
        )
    )

    # Rail side surfaces make thickness/rocker changes visually real.
    for sgn in (-1, 1):
        rail_y = sgn * half
        zbot = np.array([bottom_z(float(xv), float(yv)) for xv, yv in zip(xs, rail_y)])
        ztop = np.array([deck_z(float(xv), float(yv)) for xv, yv in zip(xs, rail_y)])
        XR = np.vstack([xs, xs])
        YR = np.vstack([rail_y, rail_y])
        ZR = np.vstack([zbot, ztop])
        fig.add_trace(
            go.Surface(
                x=XR,
                y=YR,
                z=ZR,
                surfacecolor=np.zeros_like(XR),
                colorscale=[[0, "#0a1118"], [1, "#13242f"]],
                showscale=False,
                opacity=1.0,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=rail_y,
                z=ztop,
                mode="lines",
                line=dict(color="#071018", width=5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    center_z = np.array([deck_z(float(xv), 0.0) for xv in xs])
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=np.zeros_like(xs),
            z=center_z + 0.0012,
            mode="lines",
            line=dict(color="#7ee7ff", width=4),
            hoverinfo="skip",
            name="Deck centreline",
        )
    )

    # Calculated wetted footprint overlaid on the actual parametric bottom mesh.
    wet_mask = (X <= wet) & (Zbottom <= 0.005)
    Zw = np.where(wet_mask, Zbottom + 0.0014, np.nan)
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Zw,
            surfacecolor=np.where(wet_mask, 1.0, np.nan),
            colorscale=[[0, "#ff6a13"], [1, "#ffc15a"]],
            showscale=False,
            opacity=0.88,
            hovertemplate="Predicted wetted footprint<extra></extra>",
            name="Wetted footprint",
        )
    )
    wet_half = half_width_at(wet)
    contact_y = np.linspace(-0.90 * wet_half, 0.90 * wet_half, 25)
    contact_z = np.array([bottom_z(wet, float(yy)) + 0.002 for yy in contact_y])
    fig.add_trace(
        go.Scatter3d(
            x=[wet] * len(contact_y),
            y=contact_y,
            z=contact_z,
            mode="lines",
            line=dict(color="#ffe08a", width=6),
            hovertemplate=f"Predicted wet-front x={wet:.3f} m<extra></extra>",
            name="Wet-front line",
        )
    )

    # Calculated longitudinal reference points.
    lcg_z = deck_z(min(lcg, length), 0.0) + 0.025
    lcp_z = bottom_z(min(lcp, length), 0.0) + 0.006
    fig.add_trace(
        go.Scatter3d(
            x=[lcg, lcp],
            y=[0, 0],
            z=[lcg_z, lcp_z],
            mode="markers+text",
            text=["LCG", "LCP"],
            textposition="top center",
            marker=dict(
                size=[8, 8],
                color=["#ff4d82", "#7df58a"],
                line=dict(width=1, color="#071018"),
            ),
            textfont=dict(size=12),
            name="LCG / LCP",
            hovertemplate="%{text}<br>x=%{x:.3f} m<extra></extra>",
        )
    )

    # Tow direction.
    anchor_x = float(np.clip(lcg + 0.06 * length, 0.0, length))
    anchor_z = deck_z(anchor_x, 0.0) + 0.16
    tow_abs = math.radians(trim_deg + tow_angle_deg)
    tow_length = 0.44 * length
    tx = anchor_x + tow_length * math.cos(tow_abs)
    tz = anchor_z + tow_length * math.sin(tow_abs)
    fig.add_trace(
        go.Scatter3d(
            x=[anchor_x, tx],
            y=[0, 0],
            z=[anchor_z, tz],
            mode="lines",
            line=dict(color="#ff315f", width=7),
            name="Tow direction",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[tx],
            y=[0],
            z=[tz],
            u=[math.cos(tow_abs)],
            v=[0],
            w=[math.sin(tow_abs)],
            sizemode="absolute",
            sizeref=0.07 * length,
            anchor="tip",
            colorscale=[[0, "#ff315f"], [1, "#ff315f"]],
            showscale=False,
            hovertemplate=(
                f"Tow direction<br>{tow_force_n:.0f} N<extra></extra>"
                if tow_force_n
                else "Tow direction<extra></extra>"
            ),
            name="Tow vector",
        )
    )

    # Travel direction.
    arrow_y = -0.72 * beam
    arrow_z = 0.032
    fig.add_trace(
        go.Scatter3d(
            x=[-0.08 * length, 0.14 * length],
            y=[arrow_y, arrow_y],
            z=[arrow_z, arrow_z],
            mode="lines",
            line=dict(color="#5cddff", width=6),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[0.14 * length],
            y=[arrow_y],
            z=[arrow_z],
            u=[1],
            v=[0],
            w=[0],
            sizemode="absolute",
            sizeref=0.06 * length,
            anchor="tip",
            colorscale=[[0, "#5cddff"], [1, "#5cddff"]],
            showscale=False,
            hovertemplate="Travel direction<extra></extra>",
        )
    )

    if show_forces:
        fref = max(
            float(weight_n or 1.0),
            float(lift_n or 1.0),
            float(tow_force_n or 1.0),
            1.0,
        )
        base_len = min(0.18 * length, 0.30)

        def add_force_arrow(x0, y0, z0, dx, dy, dz, force, label, color):
            mag = max(float(force or 0.0), 0.0)
            if mag <= 0:
                return
            length_scale = base_len * (0.35 + 0.65 * math.sqrt(mag / fref))
            norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            dxn, dyn, dzn = dx / norm, dy / norm, dz / norm
            x1 = x0 + dxn * length_scale
            y1 = y0 + dyn * length_scale
            z1 = z0 + dzn * length_scale
            fig.add_trace(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode="lines",
                    line=dict(color=color, width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Cone(
                    x=[x1],
                    y=[y1],
                    z=[z1],
                    u=[dxn],
                    v=[dyn],
                    w=[dzn],
                    sizemode="absolute",
                    sizeref=0.045 * length,
                    anchor="tip",
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    hovertemplate=f"{label}<br>{mag:.0f} N<extra></extra>",
                    name=label,
                )
            )

        add_force_arrow(
            lcg, 0.02 * beam, lcg_z, 0, 0, -1, weight_n, "Weight", "#ffd166"
        )
        hydro_dx = -(
            float(water_drag_n or 0.0) / max(float(lift_n or 1.0), 1.0)
        )
        add_force_arrow(
            lcp,
            -0.02 * beam,
            lcp_z,
            hydro_dx,
            0,
            1,
            lift_n,
            "Hydrodynamic resultant",
            "#67f08a",
        )
        add_force_arrow(
            lcg,
            0.08 * beam,
            lcg_z + 0.02,
            -1,
            0,
            0,
            aero_drag_n,
            "Rider aero drag",
            "#b79cff",
        )

    fig.update_layout(
        height=640,
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="x from tail (m)",
                showbackground=False,
                gridcolor="rgba(128,128,128,.16)",
            ),
            yaxis=dict(
                title="beam (m)",
                showbackground=False,
                gridcolor="rgba(128,128,128,.16)",
            ),
            zaxis=dict(
                title="z (m) — true scale",
                showbackground=False,
                gridcolor="rgba(128,128,128,.16)",
            ),
            aspectmode="data",
            camera=_camera_for(view_name),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision=f"speed-planing-3d-{view_name}",
    )
    return fig


def swatch(color, label, value=None):
    suffix = f" <b>{value}</b>" if value else ""
    return (
        f'<div style="display:flex;align-items:center;gap:.55rem;margin:.22rem 0;">'
        f'<span style="width:16px;height:5px;border-radius:4px;background:{color};display:inline-block;"></span>'
        f"<span>{label}{suffix}</span></div>"
    )


st.title("Speed Planing Lab")
st.caption("Interactive high-speed planing explorer · OpenPlaning-ready · CFD-ready")

with st.sidebar:
    st.header("Operating point")
    preset = st.selectbox(
        "Preset",
        ["160 km/h record board", "Conventional planing reference"],
    )
    if preset == "160 km/h record board":
        defaults = dict(
            speed=160.0,
            mass=90.0,
            beam=0.28,
            lcg=0.30,
            beta=0.0,
            tow=5.0,
            cda=0.32,
            length=1.72,
            tail_ratio=0.82,
            max_beam_pos=0.42,
            taper_start=0.68,
            tip_ratio=0.12,
            nose_rocker=0.035,
            rocker_start=0.73,
            tail_rocker=0.0,
            thickness=0.018,
            crown=0.004,
        )
    else:
        defaults = dict(
            speed=45.0,
            mass=900.0,
            beam=2.0,
            lcg=1.5,
            beta=10.0,
            tow=0.0,
            cda=0.0,
            length=5.0,
            tail_ratio=0.85,
            max_beam_pos=0.35,
            taper_start=0.72,
            tip_ratio=0.20,
            nose_rocker=0.20,
            rocker_start=0.68,
            tail_rocker=0.03,
            thickness=0.12,
            crown=0.025,
        )

    speed = st.slider("Speed (km/h)", 20.0, 220.0, defaults["speed"], 2.0)
    mass = st.number_input(
        "Mass incl. equipment (kg)", 10.0, 10000.0, defaults["mass"], 1.0
    )
    beam = st.number_input("Beam (m)", 0.08, 10.0, defaults["beam"], 0.01)
    lcg = st.number_input(
        "LCG from tail/stern (m)", 0.02, 10.0, defaults["lcg"], 0.01
    )
    beta = st.slider("Deadrise (deg)", 0.0, 30.0, defaults["beta"], 0.5)
    tow = st.slider(
        "Tow angle relative to keel (deg)", -10.0, 25.0, defaults["tow"], 0.5
    )
    cda = st.number_input("Rider CdA (m²)", 0.0, 3.0, defaults["cda"], 0.01)

    with st.expander("Parametric board geometry", expanded=True):
        length = st.slider("Board length (m)", 1.20, 5.50, defaults["length"], 0.02)
        tail_width_ratio = st.slider(
            "Tail width / max beam", 0.45, 1.00, defaults["tail_ratio"], 0.01
        )
        max_beam_pos = st.slider(
            "Max-beam position (% length from tail)",
            15,
            65,
            int(round(defaults["max_beam_pos"] * 100)),
            1,
        ) / 100.0
        taper_start = st.slider(
            "Nose taper starts (% length from tail)",
            50,
            90,
            int(round(defaults["taper_start"] * 100)),
            1,
        ) / 100.0
        if taper_start <= max_beam_pos + 0.03:
            taper_start = min(max_beam_pos + 0.03, 0.97)
            st.caption("Nose taper start automatically kept aft of max-beam station.")
        tip_width_ratio = st.slider(
            "Tip width / max beam", 0.03, 0.45, defaults["tip_ratio"], 0.01
        )
        nose_rocker = (
            st.slider(
                "Nose rocker (mm)",
                0,
                250,
                int(round(defaults["nose_rocker"] * 1000)),
                2,
            )
            / 1000.0
        )
        rocker_start = st.slider(
            "Nose rocker starts (% length from tail)",
            50,
            92,
            int(round(defaults["rocker_start"] * 100)),
            1,
        ) / 100.0
        tail_rocker = (
            st.slider(
                "Tail rocker (mm)",
                0,
                80,
                int(round(defaults["tail_rocker"] * 1000)),
                1,
            )
            / 1000.0
        )
        thickness = (
            st.slider(
                "Board thickness (mm)",
                8,
                160,
                int(round(defaults["thickness"] * 1000)),
                1,
            )
            / 1000.0
        )
        deck_crown_height = (
            st.slider(
                "Deck crown (mm)",
                0,
                40,
                int(round(defaults["crown"] * 1000)),
                1,
            )
            / 1000.0
        )
        st.caption(
            "These controls change the actual 3D mesh. Beam and deadrise also enter "
            "the analytical solver; the other shape variables become hydrodynamic "
            "design variables once the CFD/surrogate model is connected."
        )
        if lcg >= length:
            st.warning("LCG is at or beyond the board nose; increase length or move LCG aft.")

    with st.expander("Fluid / surface"):
        rho_w = st.number_input("Water density (kg/m³)", 900.0, 1100.0, 1000.0, 1.0)
        nu_w = st.number_input("Water ν (m²/s)", value=1.0e-6, format="%.2e")
        rho_air = st.number_input("Air density (kg/m³)", 0.5, 2.0, 1.225, 0.01)
        rough_um = st.number_input("Surface roughness (µm)", 0.0, 1000.0, 5.0, 1.0)

params = dict(
    speed_kmh=speed,
    mass=mass,
    beam=beam,
    lcg=lcg,
    beta=beta,
    tow_angle=tow,
    cda=cda,
    rho_w=rho_w,
    nu_w=nu_w,
    rho_air=rho_air,
    rough_um=rough_um,
)

try:
    r = evaluate(**params)
except Exception as e:
    st.error(f"Model solve failed: {e}")
    st.stop()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Trim", f"{r['trim_deg']:.2f}°")
m2.metric("Tow force", f"{r['tow_force_n'] / 1000:.2f} kN")
m3.metric("Water drag", f"{r['water_resistance_n']:.0f} N")
m4.metric("Wetted length", f"{r['wetted_length_m'] * 1000:.0f} mm")
m5.metric("FnB", f"{r['fn_b']:.1f}")

score = r["validity_score"]
if score == 100:
    st.success(
        f"Analytical correlation coordinates in displayed range · validity score {score}%"
    )
elif score >= 60:
    st.warning(f"Partly extrapolated analytical model · validity score {score}%")
else:
    st.error(
        f"Strong analytical extrapolation · validity score {score}% — use for screening, not final design"
    )

with st.expander("Validity details", expanded=score < 100):
    for name, ok in r["checks"].items():
        st.write(("✅ " if ok else "⚠️ ") + name)
    if HAVE_OPENPLANING:
        st.caption(
            "OpenPlaning 0.4.8 is installed. The compact screening solver remains "
            "active while the extreme-regime adapter is validated."
        )
    else:
        st.caption(
            "OpenPlaning package not available; compact Savitsky-family screening solver active."
        )

st.subheader("3D running geometry")
view_col, info_col = st.columns([2.55, 1])
with view_col:
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        view_name = st.selectbox(
            "Camera", ["Perspective", "Top", "Side", "Chase"], index=0
        )
    with c2:
        show_forces = st.toggle("Force vectors", value=True)

    fig3d = make_board_3d(
        length=length,
        beam=beam,
        trim_deg=r["trim_deg"],
        beta_deg=beta,
        wet_len=r["wetted_length_m"],
        lcg=lcg,
        lcp=r["lcp_m"],
        tow_angle_deg=tow,
        tail_width_ratio=tail_width_ratio,
        max_beam_pos=max_beam_pos,
        taper_start=taper_start,
        tip_width_ratio=tip_width_ratio,
        nose_rocker=nose_rocker,
        rocker_start=rocker_start,
        tail_rocker=tail_rocker,
        thickness=thickness,
        deck_crown_height=deck_crown_height,
        view_name=view_name,
        show_forces=show_forces,
        lift_n=r["lift_n"],
        weight_n=r["weight_n"],
        water_drag_n=r["water_resistance_n"],
        aero_drag_n=r["aero_drag_n"],
        tow_force_n=r["tow_force_n"],
    )
    st.plotly_chart(
        fig3d,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.caption(
        "True physical geometry scale. The displayed board body is generated directly "
        "from the parametric geometry controls. Orange = predicted wetted footprint; "
        "pale yellow = wet-front line. No synthetic spray or wake is shown."
    )

with info_col:
    st.markdown("#### Arrow legend")
    legend_html = ""
    if show_forces:
        legend_html += swatch("#ffd166", "Weight ↓", f"{r['weight_n']:.0f} N")
        legend_html += swatch(
            "#67f08a", "Hydrodynamic resultant ↖", f"lift {r['lift_n']:.0f} N"
        )
        legend_html += swatch(
            "#b79cff", "Rider aero drag ←", f"{r['aero_drag_n']:.0f} N"
        )
    legend_html += swatch("#ff315f", "Tow direction ↗", f"{r['tow_force_n']:.0f} N")
    legend_html += swatch("#5cddff", "Travel direction →")
    st.markdown(legend_html, unsafe_allow_html=True)
    st.caption(
        "Arrow directions are physical/model-based. Force-arrow lengths are normalized "
        "so they remain visible in the 3D scene."
    )

    st.markdown("#### Geometry")
    st.metric("Board length", f"{length:.2f} m")
    st.metric("Max beam", f"{beam * 1000:.0f} mm")
    st.metric("Tail width", f"{beam * tail_width_ratio * 1000:.0f} mm")
    st.metric("Tip width", f"{beam * tip_width_ratio * 1000:.0f} mm")
    st.metric("Nose rocker", f"{nose_rocker * 1000:.0f} mm")

    st.markdown("#### Running state")
    st.metric("Predicted wetted area", f"{r['wetted_area_m2']:.3f} m²")
    st.metric(
        "Wetted / board length",
        f"{100 * min(r['wetted_length_m'] / length, 1.0):.1f}%",
    )
    st.metric("LCP from tail", f"{r['lcp_m'] * 1000:.0f} mm")
    st.metric("LCG from tail", f"{lcg * 1000:.0f} mm")
    st.metric("Load coefficient CL", f"{r['cl']:.4f}")

    st.markdown("#### Model coupling")
    st.caption(
        "Currently coupled to analytical hydrodynamics: beam, deadrise, LCG and operating "
        "point. Geometry-only for now: length, tail/tip widths, taper stations, rocker, "
        "thickness and deck crown. The CFD surrogate can consume all of them later."
    )

    st.markdown("#### Load breakdown")
    st.dataframe(
        pd.DataFrame(
            {
                "Component": [
                    "Friction",
                    "Pressure/form",
                    "Water total",
                    "Rider aero",
                    "Tow tension",
                ],
                "Force (N)": [
                    r["friction_n"],
                    r["pressure_n"],
                    r["water_resistance_n"],
                    r["aero_drag_n"],
                    r["tow_force_n"],
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.subheader("Speed sweep")
vmin, vmax, npts = st.columns(3)
with vmin:
    smin = st.number_input(
        "Min km/h", 20.0, 220.0, max(20.0, speed - 60), 5.0
    )
with vmax:
    smax = st.number_input(
        "Max km/h", 30.0, 260.0, min(220.0, speed + 20), 5.0
    )
with npts:
    n = st.slider("Points", 5, 25, 13)

rows = []
if smax > smin:
    for vv in np.linspace(smin, smax, n):
        try:
            rr = evaluate(**{**params, "speed_kmh": float(vv)})
            rows.append(
                {
                    "Speed km/h": vv,
                    "Tow N": rr["tow_force_n"],
                    "Water N": rr["water_resistance_n"],
                    "Aero N": rr["aero_drag_n"],
                    "Trim deg": rr["trim_deg"],
                    "Wet m": rr["wetted_length_m"],
                    "Validity %": rr["validity_score"],
                }
            )
        except Exception:
            pass

if rows:
    df = pd.DataFrame(rows)
    f = go.Figure()
    for col in ["Tow N", "Water N", "Aero N"]:
        f.add_trace(
            go.Scatter(
                x=df["Speed km/h"],
                y=df[col],
                mode="lines+markers",
                name=col,
            )
        )
    f.update_layout(
        height=390,
        xaxis_title="Speed (km/h)",
        yaxis_title="Force (N)",
    )
    st.plotly_chart(f, use_container_width=True)
    st.download_button(
        "Download sweep CSV",
        df.to_csv(index=False),
        "speed_planing_sweep.csv",
        "text/csv",
    )

st.divider()
st.subheader("Beam × LCG design search")
st.caption(
    "Small deterministic analytical search for exploration; the full geometry "
    "parameter set becomes optimizable when the CFD surrogate is connected."
)
b1, b2, x1, x2, gn = st.columns(5)
with b1:
    blo = st.number_input(
        "Beam min", 0.08, 2.0, max(0.08, beam * 0.85), 0.01
    )
with b2:
    bhi = st.number_input("Beam max", 0.09, 2.0, beam * 1.15, 0.01)
with x1:
    xlo = st.number_input(
        "LCG min", 0.02, 3.0, max(0.02, lcg * 0.8), 0.01
    )
with x2:
    xhi = st.number_input("LCG max", 0.03, 3.0, lcg * 1.2, 0.01)
with gn:
    grid = st.slider("Grid", 3, 9, 5)

if st.button("Run design search"):
    out = []
    for bb in np.linspace(blo, bhi, grid):
        for xx in np.linspace(xlo, xhi, grid):
            try:
                rr = evaluate(**{**params, "beam": float(bb), "lcg": float(xx)})
                out.append(
                    {
                        "Beam m": bb,
                        "LCG m": xx,
                        "Tow N": rr["tow_force_n"],
                        "Water N": rr["water_resistance_n"],
                        "Trim deg": rr["trim_deg"],
                        "Validity %": rr["validity_score"],
                    }
                )
            except Exception:
                pass
    if out:
        odf = pd.DataFrame(out).sort_values("Tow N")
        best = odf.iloc[0]
        st.success(
            f"Best in this grid: beam {best['Beam m'] * 1000:.0f} mm · "
            f"LCG {best['LCG m'] * 1000:.0f} mm · tow {best['Tow N']:.0f} N"
        )
        st.dataframe(odf.head(10), hide_index=True, use_container_width=True)

st.divider()
st.info(
    "CFD integration point: the future surrogate should replace/calibrate the load "
    "and equilibrium evaluator and accept the full parametric board geometry. "
    "Extreme 160 km/h analytical results are screening estimates only."
)
