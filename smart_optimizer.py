import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution


def _scenario_speeds(mode, current_speed, range_min, range_max, range_points):
    if mode == "Current speed":
        return [float(current_speed)]
    return [float(v) for v in np.linspace(range_min, range_max, range_points)]


def _constraint_violations(states, board_length, angle_min, angle_max, max_contact_fraction, min_range_score):
    violations = []
    for state in states:
        angle = float(state["trim_deg"])
        wet_fraction = float(state["wetted_length_m"]) / max(board_length, 1e-9)
        score = float(state["validity_score"])
        if angle < angle_min:
            violations.append((angle_min - angle) / max(angle_min, 1.0))
        if angle > angle_max:
            violations.append((angle - angle_max) / max(angle_max, 1.0))
        if wet_fraction > max_contact_fraction:
            violations.append((wet_fraction - max_contact_fraction) / max(max_contact_fraction, 0.05))
        if score < min_range_score:
            violations.append((min_range_score - score) / 100.0)
    return violations


def _summarize_candidate(x, evaluate, base_params, speeds, board_length, objective_mode,
                         angle_min, angle_max, max_contact_fraction, min_range_score):
    width, center_of_gravity, bottom_v, tow_angle = [float(v) for v in x]
    if center_of_gravity <= 0 or center_of_gravity >= 0.92 * board_length:
        return None

    states = []
    try:
        for speed in speeds:
            state = evaluate(
                **{
                    **base_params,
                    "speed_kmh": speed,
                    "beam": width,
                    "lcg": center_of_gravity,
                    "beta": bottom_v,
                    "tow_angle": tow_angle,
                }
            )
            states.append(state)
    except Exception:
        return None

    violations = _constraint_violations(
        states,
        board_length,
        angle_min,
        angle_max,
        max_contact_fraction,
        min_range_score,
    )
    tow_forces = np.array([float(s["tow_force_n"]) for s in states])
    water_drag = np.array([float(s["water_resistance_n"]) for s in states])
    wet_fraction = np.array([float(s["wetted_length_m"]) / board_length for s in states])
    angles = np.array([float(s["trim_deg"]) for s in states])
    scores = np.array([float(s["validity_score"]) for s in states])

    if objective_mode == "Lowest worst-case tow force":
        objective = float(np.max(tow_forces))
    else:
        objective = float(np.mean(tow_forces))

    return {
        "width": width,
        "center_of_gravity": center_of_gravity,
        "bottom_v": bottom_v,
        "tow_angle": tow_angle,
        "objective": objective,
        "worst_tow": float(np.max(tow_forces)),
        "mean_tow": float(np.mean(tow_forces)),
        "max_water_drag": float(np.max(water_drag)),
        "max_wet_fraction": float(np.max(wet_fraction)),
        "min_angle": float(np.min(angles)),
        "max_angle": float(np.max(angles)),
        "min_range_score": float(np.min(scores)),
        "violations": violations,
        "states": states,
    }


def _pareto_mask(tow_force, wet_fraction):
    order = np.argsort(tow_force)
    mask = np.zeros(len(tow_force), dtype=bool)
    best_wet = math.inf
    for idx in order:
        if wet_fraction[idx] < best_wet - 1e-12:
            mask[idx] = True
            best_wet = wet_fraction[idx]
    return mask


def render_smart_optimizer(*, evaluate, base_params, board_length, current_width,
                           current_center_of_gravity, current_bottom_v, current_tow_angle):
    st.divider()
    st.subheader("Legacy fast-model optimizer (secondary)")
    st.caption(
        "Searches the legacy fast model only. At record speed it is a trend explorer, not the V1 design authority; "
        "the published-experiment synthesis above takes precedence because the pitch closure is incomplete."
    )

    try:
        current_state = evaluate(**base_params)
        current_range_score = int(current_state["validity_score"])
    except Exception:
        current_range_score = 0

    mode_col, goal_col, quality_col = st.columns(3)
    with mode_col:
        scenario_mode = st.selectbox(
            "Optimize for",
            ["Current speed", "Speed range"],
            key="smart_opt_scenario_mode",
        )
    with goal_col:
        objective_mode = st.selectbox(
            "Goal",
            ["Lowest worst-case tow force", "Lowest average tow force"],
            key="smart_opt_objective_mode",
        )
    with quality_col:
        search_quality = st.selectbox(
            "Search depth",
            ["Fast", "Thorough"],
            key="smart_opt_quality",
        )

    current_speed = float(base_params["speed_kmh"])
    if current_speed >= 100.0:
        st.warning("Do not use its optimum as the record-board geometry. The high-speed V1 above is anchored to published flat-plate evidence; this optimizer still uses the incomplete legacy pitch equilibrium.")
    if scenario_mode == "Speed range":
        s1, s2, s3 = st.columns(3)
        with s1:
            range_min = st.number_input(
                "Lowest speed (km/h)",
                20.0,
                220.0,
                max(20.0, current_speed - 40.0),
                5.0,
                key="smart_opt_speed_min",
            )
        with s2:
            range_max = st.number_input(
                "Highest speed (km/h)",
                30.0,
                240.0,
                current_speed,
                5.0,
                key="smart_opt_speed_max",
            )
        with s3:
            range_points = st.slider(
                "Speeds checked", 3, 7, 4, 1, key="smart_opt_speed_points"
            )
        if range_max <= range_min:
            st.warning("Highest speed must be above lowest speed.")
            return
    else:
        range_min = range_max = current_speed
        range_points = 1

    speeds = _scenario_speeds(
        scenario_mode, current_speed, range_min, range_max, range_points
    )

    with st.expander("Search limits", expanded=True):
        b1, b2, x1, x2 = st.columns(4)
        with b1:
            width_min = st.number_input(
                "Board width min (m)",
                0.08,
                2.5,
                max(0.08, current_width * 0.75),
                0.01,
                key="smart_opt_width_min",
            )
        with b2:
            width_max = st.number_input(
                "Board width max (m)",
                0.09,
                3.0,
                min(3.0, current_width * 1.30),
                0.01,
                key="smart_opt_width_max",
            )
        with x1:
            cg_min = st.number_input(
                "Center of gravity min from tail (m)",
                0.02,
                max(0.03, board_length * 0.9),
                max(0.02, current_center_of_gravity * 0.70),
                0.01,
                key="smart_opt_cg_min",
            )
        with x2:
            cg_default_max = min(board_length * 0.88, current_center_of_gravity * 1.35)
            cg_max = st.number_input(
                "Center of gravity max from tail (m)",
                0.03,
                max(0.04, board_length * 0.92),
                max(0.03, cg_default_max),
                0.01,
                key="smart_opt_cg_max",
            )

        v1, v2, t1, t2 = st.columns(4)
        with v1:
            v_min = st.number_input(
                "Bottom V min (deg)", 0.0, 25.0, 0.0, 0.5, key="smart_opt_v_min"
            )
        with v2:
            v_max = st.number_input(
                "Bottom V max (deg)",
                0.0,
                30.0,
                max(5.0, current_bottom_v + 5.0),
                0.5,
                key="smart_opt_v_max",
            )
        with t1:
            tow_min = st.number_input(
                "Tow-line angle min (deg)",
                -10.0,
                20.0,
                max(-10.0, current_tow_angle - 5.0),
                0.5,
                key="smart_opt_tow_min",
            )
        with t2:
            tow_max = st.number_input(
                "Tow-line angle max (deg)",
                -5.0,
                25.0,
                min(25.0, current_tow_angle + 7.0),
                0.5,
                key="smart_opt_tow_max",
            )

    with st.expander("Keep the result sensible", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        high_speed = max(speeds) >= 100.0
        with c1:
            angle_min = st.number_input(
                "Minimum board angle (deg)",
                0.0,
                10.0,
                2.0 if high_speed else 1.5,
                0.1,
                key="smart_opt_angle_min",
            )
        with c2:
            angle_max = st.number_input(
                "Maximum board angle (deg)",
                0.5,
                15.0,
                4.0 if high_speed else 10.0,
                0.1,
                key="smart_opt_angle_max",
            )
        with c3:
            max_contact_percent = st.slider(
                "Maximum board length touching water (%)",
                10,
                90,
                50,
                5,
                key="smart_opt_contact_max",
            )
        with c4:
            min_range_score = st.slider(
                "Minimum model-range score (%)",
                0,
                100,
                int(max(0, min(100, current_range_score))),
                20,
                key="smart_opt_range_min",
            )
        st.caption(
            "The model-range score prevents the optimizer from winning simply by moving even farther outside the model's usual operating range."
        )

    invalid_bounds = (
        width_max <= width_min
        or cg_max <= cg_min
        or v_max < v_min
        or tow_max <= tow_min
        or angle_max <= angle_min
    )
    if invalid_bounds:
        st.warning("One or more maximum limits must be above the corresponding minimum.")
        return

    run_col, explore_col = st.columns(2)
    with run_col:
        run_optimizer = st.button(
            "Find best setup", type="primary", key="smart_opt_run"
        )
    with explore_col:
        exploration_count = st.select_slider(
            "Designs in exploration view",
            options=[100, 200, 300, 500],
            value=200,
            key="smart_opt_explore_count",
        )

    if run_optimizer:
        bounds = [
            (float(width_min), float(width_max)),
            (float(cg_min), float(cg_max)),
            (float(v_min), float(v_max)),
            (float(tow_min), float(tow_max)),
        ]
        maxiter, popsize = ((18, 7) if search_quality == "Fast" else (35, 10))
        max_contact_fraction = max_contact_percent / 100.0

        def objective(x):
            summary = _summarize_candidate(
                x,
                evaluate,
                base_params,
                speeds,
                board_length,
                objective_mode,
                angle_min,
                angle_max,
                max_contact_fraction,
                min_range_score,
            )
            if summary is None:
                return 1e9
            penalty = 1e6 * sum(v * v for v in summary["violations"])
            return summary["objective"] + penalty

        with st.spinner("Searching the setup space..."):
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                tol=2e-4,
                polish=True,
                updating="immediate",
                workers=1,
            )
            best = _summarize_candidate(
                result.x,
                evaluate,
                base_params,
                speeds,
                board_length,
                objective_mode,
                angle_min,
                angle_max,
                max_contact_fraction,
                min_range_score,
            )

            rng = np.random.default_rng(20260810)
            samples = rng.uniform(
                low=np.array([b[0] for b in bounds]),
                high=np.array([b[1] for b in bounds]),
                size=(int(exploration_count), 4),
            )
            if best is not None:
                samples = np.vstack([samples, result.x])

            exploration = []
            for x in samples:
                summary = _summarize_candidate(
                    x,
                    evaluate,
                    base_params,
                    speeds,
                    board_length,
                    objective_mode,
                    angle_min,
                    angle_max,
                    max_contact_fraction,
                    min_range_score,
                )
                if summary is not None and not summary["violations"]:
                    exploration.append(summary)

        st.session_state["smart_optimizer_result"] = {
            "best": best,
            "exploration": exploration,
            "speeds": speeds,
            "objective_mode": objective_mode,
            "board_length": board_length,
        }

    saved = st.session_state.get("smart_optimizer_result")
    if not saved:
        return

    best = saved.get("best")
    exploration = saved.get("exploration", [])
    saved_speeds = saved.get("speeds", speeds)
    if best is None:
        st.error("No setup satisfying these limits was found. Widen the search limits or relax one of the sensible-result limits.")
        return
    if best["violations"]:
        st.warning("The optimizer found a low-force setup but could not fully satisfy every limit. Relax a limit or widen the search space.")

    st.markdown("#### Best setup found")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Board width", f"{best['width'] * 1000:.0f} mm")
    m2.metric("Center of gravity from tail", f"{best['center_of_gravity'] * 1000:.0f} mm")
    m3.metric("Bottom V angle", f"{best['bottom_v']:.1f}°")
    m4.metric("Tow-line angle", f"{best['tow_angle']:.1f}°")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Worst tow force", f"{best['worst_tow']:.0f} N")
    p2.metric("Average tow force", f"{best['mean_tow']:.0f} N")
    p3.metric("Maximum water contact", f"{100 * best['max_wet_fraction']:.1f}% of board")
    p4.metric("Lowest model-range score", f"{best['min_range_score']:.0f}%")

    scenario_rows = []
    for speed, state in zip(saved_speeds, best["states"]):
        scenario_rows.append(
            {
                "Speed (km/h)": speed,
                "Tow force (N)": state["tow_force_n"],
                "Water drag (N)": state["water_resistance_n"],
                "Board angle (deg)": state["trim_deg"],
                "Board touching water (mm)": state["wetted_length_m"] * 1000,
                "Model-range score (%)": state["validity_score"],
            }
        )
    st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, use_container_width=True)

    if exploration:
        explore_df = pd.DataFrame(
            {
                "Tow force (N)": [e["objective"] for e in exploration],
                "Maximum water contact (%)": [100 * e["max_wet_fraction"] for e in exploration],
                "Model-range score (%)": [e["min_range_score"] for e in exploration],
                "Board width (mm)": [1000 * e["width"] for e in exploration],
                "Center of gravity from tail (mm)": [1000 * e["center_of_gravity"] for e in exploration],
                "Bottom V angle (deg)": [e["bottom_v"] for e in exploration],
                "Tow-line angle (deg)": [e["tow_angle"] for e in exploration],
            }
        )
        tow_values = explore_df["Tow force (N)"].to_numpy()
        wet_values = explore_df["Maximum water contact (%)"].to_numpy()
        pareto = _pareto_mask(tow_values, wet_values)

        st.markdown("#### Explore the good designs")
        st.caption(
            "Lower-left is better: less tow force and less board touching the water. The highlighted line is the best trade-off set — improving one of those two measures would worsen the other."
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tow_values,
                y=wet_values,
                mode="markers",
                name="Feasible designs",
                marker=dict(
                    size=8,
                    opacity=0.58,
                    color=explore_df["Model-range score (%)"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Model range %"),
                ),
                customdata=explore_df[
                    [
                        "Board width (mm)",
                        "Center of gravity from tail (mm)",
                        "Bottom V angle (deg)",
                        "Tow-line angle (deg)",
                    ]
                ],
                hovertemplate=(
                    "Tow force %{x:.0f} N<br>Water contact %{y:.1f}%<br>"
                    "Width %{customdata[0]:.0f} mm<br>Center of gravity %{customdata[1]:.0f} mm from tail<br>"
                    "Bottom V %{customdata[2]:.1f}°<br>Tow line %{customdata[3]:.1f}°<extra></extra>"
                ),
            )
        )
        if np.any(pareto):
            pf = explore_df.loc[pareto].sort_values("Tow force (N)")
            fig.add_trace(
                go.Scatter(
                    x=pf["Tow force (N)"],
                    y=pf["Maximum water contact (%)"],
                    mode="lines+markers",
                    name="Best trade-off line",
                    line=dict(width=3),
                    marker=dict(size=10),
                )
            )
        fig.update_layout(
            height=470,
            xaxis_title="Tow force (N)",
            yaxis_title="Maximum board length touching water (%)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        top = explore_df.sort_values("Tow force (N)").head(12)
        st.dataframe(top, hide_index=True, use_container_width=True)
        st.download_button(
            "Download optimizer exploration CSV",
            explore_df.to_csv(index=False),
            "speed_planing_optimizer_exploration.csv",
            "text/csv",
            key="smart_opt_download",
        )
