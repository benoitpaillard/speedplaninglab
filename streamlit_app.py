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
    cl0 = tau_deg**1.1 * (0.012 * lam**0.5 + 0.0055 * lam**2.5 / max(fn_b**2, 1e-9))
    return cl0 - 0.0065 * beta_deg * max(cl0, 1e-12)**0.6


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
        lcp = lam * beam * (0.75 - 1.0 / (5.21 * (fn_b / max(lam, 1e-9)) ** 2 + 2.39))
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
        "checks": checks,
        "validity_score": score,
    }


def evaluate(**p):
    return screening(**p)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def board_half_width(x, length, beam):
    """Directional high-speed planform used only for visualisation."""
    u = np.asarray(x) / max(length, 1e-9)
    aft = 0.82 + 0.18 * np.sin(np.clip(u / 0.62, 0, 1) * np.pi / 2)
    nose_taper = 1.0 - 0.88 * _smoothstep((u - 0.62) / 0.38)
    width_factor = np.where(u <= 0.62, aft, nose_taper)
    return 0.5 * beam * np.maximum(width_factor, 0.10)


def make_board_3d(length, beam, trim_deg, beta_deg, wet_len, lcg, lcp, tow_angle_deg,
                  vertical_scale=4.0, nose_rocker=0.035, thickness=0.018):
    """Create an orbitable Plotly 3D rendering of the running board geometry."""
    nx, ny = 72, 31
    xs = np.linspace(0.0, length, nx)
    yn = np.linspace(-1.0, 1.0, ny)
    X = np.repeat(xs[:, None], ny, axis=1)
    half = board_half_width(xs, length, beam)
    Y = half[:, None] * yn[None, :]

    tau = math.radians(trim_deg)
    beta = math.radians(beta_deg)
    u = X / max(length, 1e-9)
    rocker_s = np.clip((u - 0.73) / 0.27, 0.0, 1.0)
    intrinsic_rocker = nose_rocker * rocker_s**2.2
    deadrise = np.abs(Y) * math.tan(beta)
    Zbottom_phys = X * math.tan(tau) + intrinsic_rocker + deadrise

    deck_crown = 0.004 * (1.0 - (Y / np.maximum(half[:, None], 1e-6)) ** 2)
    Ztop_phys = Zbottom_phys + thickness + deck_crown
    Zbottom = Zbottom_phys * vertical_scale
    Ztop = Ztop_phys * vertical_scale

    fig = go.Figure()

    wx = np.linspace(-0.10 * length, 1.08 * length, 2)
    wy = np.linspace(-0.85 * beam, 0.85 * beam, 2)
    WX, WY = np.meshgrid(wx, wy)
    WZ = np.zeros_like(WX)
    fig.add_trace(go.Surface(
        x=WX, y=WY, z=WZ,
        surfacecolor=np.zeros_like(WX),
        colorscale=[[0, "rgba(25,126,185,0.34)"], [1, "rgba(25,126,185,0.34)"]],
        showscale=False, opacity=0.34, hoverinfo="skip", name="Water plane",
    ))

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zbottom, surfacecolor=X,
        colorscale=[[0, "#111827"], [0.55, "#263548"], [1, "#44566d"]],
        showscale=False, opacity=0.98,
        lighting=dict(ambient=0.45, diffuse=0.75, roughness=0.35, specular=0.6, fresnel=0.15),
        lightposition=dict(x=-2, y=-3, z=6),
        hovertemplate="Bottom<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f}<extra></extra>",
        name="Bottom",
    ))
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Ztop, surfacecolor=X,
        colorscale=[[0, "#d7dde5"], [0.60, "#f1f4f8"], [1, "#c8d0da"]],
        showscale=False, opacity=1.0,
        lighting=dict(ambient=0.55, diffuse=0.8, roughness=0.25, specular=0.8, fresnel=0.2),
        lightposition=dict(x=-2, y=-3, z=7),
        hovertemplate="Deck<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f}<extra></extra>",
        name="Deck",
    ))

    for sgn in (-1, 1):
        rail_y = sgn * half
        rail_z = (xs * math.tan(tau)
                  + nose_rocker * np.clip((xs / length - 0.73) / 0.27, 0, 1) ** 2.2
                  + np.abs(rail_y) * math.tan(beta)
                  + thickness * 0.55) * vertical_scale
        fig.add_trace(go.Scatter3d(
            x=xs, y=rail_y, z=rail_z, mode="lines",
            line=dict(color="rgba(14,20,28,0.85)", width=5),
            hoverinfo="skip", showlegend=False,
        ))

    wet = min(max(wet_len, 0.0), length)
    mask = X <= wet
    Zw = np.where(mask, Zbottom + 0.0015 * vertical_scale, np.nan)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zw,
        surfacecolor=np.where(mask, 1.0, np.nan),
        colorscale=[[0, "#ff7a18"], [1, "#ffb347"]],
        showscale=False, opacity=0.80,
        hovertemplate="Predicted wetted footprint<extra></extra>",
        name="Wetted footprint",
    ))

    def deck_center_z(xv):
        sr = max(min((xv / length - 0.73) / 0.27, 1.0), 0.0)
        return (xv * math.tan(tau) + nose_rocker * sr**2.2 + thickness + 0.004) * vertical_scale

    marker_x = [lcg, lcp]
    marker_z = [deck_center_z(lcg), deck_center_z(lcp)]
    fig.add_trace(go.Scatter3d(
        x=marker_x, y=[0, 0], z=marker_z,
        mode="markers+text", text=["CG", "LCP"], textposition="top center",
        marker=dict(size=[8, 7], color=["#e11d48", "#22c55e"], line=dict(width=1, color="white")),
        textfont=dict(size=12), name="CG / LCP",
        hovertemplate="%{text}<br>x=%{x:.3f} m<extra></extra>",
    ))

    anchor_x = lcg
    anchor_z = deck_center_z(anchor_x) + 0.18 * vertical_scale
    tow_abs = math.radians(trim_deg + tow_angle_deg)
    tow_length = 0.48 * length
    tx = anchor_x + tow_length * math.cos(tow_abs)
    tz = anchor_z + tow_length * math.sin(tow_abs) * vertical_scale
    fig.add_trace(go.Scatter3d(
        x=[anchor_x, tx], y=[0, 0], z=[anchor_z, tz],
        mode="lines", line=dict(color="#f43f5e", width=7),
        name="Tow direction", hoverinfo="skip",
    ))
    fig.add_trace(go.Cone(
        x=[tx], y=[0], z=[tz],
        u=[math.cos(tow_abs)], v=[0], w=[math.sin(tow_abs) * vertical_scale],
        sizemode="absolute", sizeref=0.08 * length, anchor="tip",
        colorscale=[[0, "#f43f5e"], [1, "#f43f5e"]],
        showscale=False, hoverinfo="skip", name="Tow vector",
    ))

    flow_z = 0.03 * vertical_scale
    fig.add_trace(go.Scatter3d(
        x=[-0.08 * length, 0.12 * length], y=[-0.58 * beam] * 2, z=[flow_z] * 2,
        mode="lines", line=dict(color="#38bdf8", width=6),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Cone(
        x=[0.12 * length], y=[-0.58 * beam], z=[flow_z],
        u=[1], v=[0], w=[0], sizemode="absolute", sizeref=0.07 * length, anchor="tip",
        colorscale=[[0, "#38bdf8"], [1, "#38bdf8"]],
        showscale=False, hovertemplate="Flow direction<extra></extra>",
    ))

    zmax = max(float(np.nanmax(Ztop)), 0.20 * vertical_scale)
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5),
        scene=dict(
            xaxis=dict(title="x from tail (m)", showbackground=False, gridcolor="rgba(128,128,128,.18)"),
            yaxis=dict(title="beam (m)", showbackground=False, gridcolor="rgba(128,128,128,.18)"),
            zaxis=dict(title="z (display scale)", showbackground=False, gridcolor="rgba(128,128,128,.18)", range=[-0.04 * vertical_scale, zmax * 1.25]),
            aspectmode="manual", aspectratio=dict(x=2.8, y=1.0, z=0.75),
            camera=dict(eye=dict(x=1.35, y=-1.55, z=0.85), center=dict(x=0.05, y=0, z=-0.08)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        uirevision="speed-planing-3d",
    )
    return fig


st.title("Speed Planing Lab")
st.caption("Interactive high-speed planing explorer · OpenPlaning-ready · CFD-ready")

with st.sidebar:
    st.header("Design")
    preset = st.selectbox("Preset", ["160 km/h record board", "Conventional planing reference"])
    if preset == "160 km/h record board":
        defaults = dict(speed=160.0, mass=90.0, beam=0.28, lcg=0.30, beta=0.0, tow=5.0, cda=0.32)
    else:
        defaults = dict(speed=45.0, mass=900.0, beam=2.0, lcg=1.5, beta=10.0, tow=0.0, cda=0.0)
    speed = st.slider("Speed (km/h)", 20.0, 220.0, defaults["speed"], 2.0)
    mass = st.number_input("Mass incl. equipment (kg)", 10.0, 10000.0, defaults["mass"], 1.0)
    beam = st.number_input("Beam (m)", 0.08, 10.0, defaults["beam"], 0.01)
    lcg = st.number_input("LCG from tail/stern (m)", 0.02, 10.0, defaults["lcg"], 0.01)
    beta = st.slider("Deadrise (deg)", 0.0, 30.0, defaults["beta"], 0.5)
    tow = st.slider("Tow angle relative to keel (deg)", -10.0, 25.0, defaults["tow"], 0.5)
    cda = st.number_input("Rider CdA (m²)", 0.0, 3.0, defaults["cda"], 0.01)
    with st.expander("Fluid / surface"):
        rho_w = st.number_input("Water density (kg/m³)", 900.0, 1100.0, 1000.0, 1.0)
        nu_w = st.number_input("Water ν (m²/s)", value=1.0e-6, format="%.2e")
        rho_air = st.number_input("Air density (kg/m³)", 0.5, 2.0, 1.225, 0.01)
        rough_um = st.number_input("Surface roughness (µm)", 0.0, 1000.0, 5.0, 1.0)

params = dict(speed_kmh=speed, mass=mass, beam=beam, lcg=lcg, beta=beta, tow_angle=tow,
              cda=cda, rho_w=rho_w, nu_w=nu_w, rho_air=rho_air, rough_um=rough_um)

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
    st.success(f"Analytical correlation coordinates in displayed range · validity score {score}%")
elif score >= 60:
    st.warning(f"Partly extrapolated analytical model · validity score {score}%")
else:
    st.error(f"Strong analytical extrapolation · validity score {score}% — use for screening, not final design")

with st.expander("Validity details", expanded=score < 100):
    for name, ok in r["checks"].items():
        st.write(("✅ " if ok else "⚠️ ") + name)
    if HAVE_OPENPLANING:
        st.caption("OpenPlaning 0.4.8 is installed on this deployment. The compact screening solver is retained as a robust baseline while the OpenPlaning adapter is validated against the extreme wakeboard regime.")
    else:
        st.caption("OpenPlaning package not available; compact Savitsky-family screening solver active.")

st.subheader("3D running geometry")
view_col, info_col = st.columns([2.5, 1])
with view_col:
    L = max(1.72, 2.2 * lcg)
    z_exaggerated = st.toggle("Vertical exaggeration ×4", value=True, help="Purely visual. Helps make trim, deadrise and rocker readable on a very thin high-speed board.")
    zscale = 4.0 if z_exaggerated else 1.0
    fig3d = make_board_3d(
        length=L, beam=beam, trim_deg=r["trim_deg"], beta_deg=beta,
        wet_len=r["wetted_length_m"], lcg=lcg, lcp=r["lcp_m"],
        tow_angle_deg=tow, vertical_scale=zscale,
    )
    st.plotly_chart(fig3d, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    st.caption("Drag to orbit · wheel/pinch to zoom · orange = predicted wetted footprint · red = tow direction. Board planform/rocker is a visual design proxy, while trim, wetted length, CG and LCP come from the active analytical model.")

with info_col:
    st.markdown("#### Running state")
    st.metric("Predicted wetted area", f"{r['wetted_area_m2']:.3f} m²")
    st.metric("LCP from tail", f"{r['lcp_m'] * 1000:.0f} mm")
    st.metric("LCG from tail", f"{lcg * 1000:.0f} mm")
    st.metric("Load coefficient CL", f"{r['cl']:.4f}")
    st.markdown("#### Load breakdown")
    st.dataframe(pd.DataFrame({
        "Component": ["Friction", "Pressure/form", "Water total", "Rider aero", "Tow tension"],
        "Force (N)": [r["friction_n"], r["pressure_n"], r["water_resistance_n"], r["aero_drag_n"], r["tow_force_n"]],
    }), hide_index=True, use_container_width=True)

st.divider()
st.subheader("Speed sweep")
vmin, vmax, npts = st.columns(3)
with vmin:
    smin = st.number_input("Min km/h", 20.0, 220.0, max(20.0, speed - 60), 5.0)
with vmax:
    smax = st.number_input("Max km/h", 30.0, 260.0, min(220.0, speed + 20), 5.0)
with npts:
    n = st.slider("Points", 5, 25, 13)
rows = []
if smax > smin:
    for vv in np.linspace(smin, smax, n):
        try:
            rr = evaluate(**{**params, "speed_kmh": float(vv)})
            rows.append({"Speed km/h": vv, "Tow N": rr["tow_force_n"], "Water N": rr["water_resistance_n"],
                         "Aero N": rr["aero_drag_n"], "Trim deg": rr["trim_deg"], "Wet m": rr["wetted_length_m"],
                         "Validity %": rr["validity_score"]})
        except Exception:
            pass
if rows:
    df = pd.DataFrame(rows)
    f = go.Figure()
    for col in ["Tow N", "Water N", "Aero N"]:
        f.add_trace(go.Scatter(x=df["Speed km/h"], y=df[col], mode="lines+markers", name=col))
    f.update_layout(height=390, xaxis_title="Speed (km/h)", yaxis_title="Force (N)")
    st.plotly_chart(f, use_container_width=True)
    st.download_button("Download sweep CSV", df.to_csv(index=False), "speed_planing_sweep.csv", "text/csv")

st.divider()
st.subheader("Beam × LCG design search")
st.caption("Small deterministic search for exploration; CFD optimization remains the high-fidelity workstream.")
b1, b2, x1, x2, gn = st.columns(5)
with b1:
    blo = st.number_input("Beam min", 0.08, 2.0, max(0.08, beam * 0.85), 0.01)
with b2:
    bhi = st.number_input("Beam max", 0.09, 2.0, beam * 1.15, 0.01)
with x1:
    xlo = st.number_input("LCG min", 0.02, 3.0, max(0.02, lcg * 0.8), 0.01)
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
                out.append({"Beam m": bb, "LCG m": xx, "Tow N": rr["tow_force_n"],
                            "Water N": rr["water_resistance_n"], "Trim deg": rr["trim_deg"],
                            "Validity %": rr["validity_score"]})
            except Exception:
                pass
    if out:
        odf = pd.DataFrame(out).sort_values("Tow N")
        best = odf.iloc[0]
        st.success(f"Best in this grid: beam {best['Beam m'] * 1000:.0f} mm · LCG {best['LCG m'] * 1000:.0f} mm · tow {best['Tow N']:.0f} N")
        st.dataframe(odf.head(10), hide_index=True, use_container_width=True)

st.divider()
st.info("CFD integration point: the future surrogate should replace/calibrate the load and equilibrium evaluator while keeping this interface and validity reporting. Extreme 160 km/h results must not be treated as final safety predictions until CFD and dynamic-stability validation are complete.")
