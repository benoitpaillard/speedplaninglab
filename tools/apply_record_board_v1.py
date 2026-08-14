from pathlib import Path


def patch_app():
    app = Path("streamlit_app.py")
    text = app.read_text()

    text = text.replace(
        'st.caption("Interactive high-speed planing explorer · OpenPlaning model · high-fidelity simulation ready")',
        'st.caption("Published-experiment synthesis · evidence-driven record board · legacy fast model for secondary trend checks")',
    )

    replacements = {
        "            beam=0.28,": "            beam=0.295,",
        "            tail_ratio=0.82,": "            tail_ratio=0.93,",
        "            max_beam_pos=0.42,": "            max_beam_pos=0.18,",
        "            taper_start=0.68,": "            taper_start=0.61,",
        "            tip_ratio=0.12,": "            tip_ratio=0.17,",
        "            rocker_start=0.73,": "            rocker_start=0.71,",
        '    beam = st.number_input("Maximum board width (m)", 0.08, 10.0, defaults["beam"], 0.01)':
            '    beam = st.number_input("Maximum board width (m)", 0.08, 10.0, defaults["beam"], 0.005)',
        '        "Center of gravity from tail (m)", 0.02, 10.0, defaults["lcg"], 0.01':
            '        "Legacy fast-model balance point from tail (m)", 0.02, 10.0, defaults["lcg"], 0.01',
    }
    for old, new in replacements.items():
        text = text.replace(old, new, 1)

    lcg_marker = '''    lcg = st.number_input(
        "Legacy fast-model balance point from tail (m)", 0.02, 10.0, defaults["lcg"], 0.01
    )
'''
    note = '    st.caption("At record speed this is only the legacy model closure variable — not a recommended physical binding or rider-CG position.")\n'
    if lcg_marker in text and note.strip() not in text:
        text = text.replace(lcg_marker, lcg_marker + note, 1)

    marker = "m1, m2, m3, m4, m5 = st.columns(5)\n"
    if "render_record_board_design(" not in text:
        if marker not in text:
            raise RuntimeError("Could not find main metric insertion point")
        addition = '''from record_board_design import render_record_board_design

render_record_board_design(
    speed_kmh=speed,
    mass_kg=mass,
    rho_w=rho_w,
    nu_w=nu_w,
)

st.divider()
st.subheader("Legacy fast-model estimate")
st.caption("Useful for trend comparison only at record speed. Its pitch closure does not include the full rider/tow-handle moment balance.")

'''
        text = text.replace(marker, addition + marker, 1)

    app.write_text(text)


def patch_optimizer():
    opt = Path("smart_optimizer.py")
    text = opt.read_text()
    text = text.replace(
        '    st.subheader("Smart board setup optimizer")',
        '    st.subheader("Legacy fast-model optimizer (secondary)")',
        1,
    )
    old_caption = (
        '        "Automatically searches board width, balance point, bottom V and tow-line angle. "\n'
        '        "It uses the same fast performance model as the dashboard, so the 160 km/h result is still an early-design estimate."'
    )
    new_caption = (
        '        "Searches the legacy fast model only. At record speed it is a trend explorer, not the V1 design authority; "\n'
        '        "the published-experiment synthesis above takes precedence because the pitch closure is incomplete."'
    )
    text = text.replace(old_caption, new_caption, 1)
    text = text.replace(
        "                0.35 if high_speed else 1.5,",
        "                2.0 if high_speed else 1.5,",
        1,
    )

    marker = '    current_speed = float(base_params["speed_kmh"])\n'
    warning = '    if current_speed >= 100.0:\n        st.warning("Do not use its optimum as the record-board geometry. The high-speed V1 above is anchored to published flat-plate evidence; this optimizer still uses the incomplete legacy pitch equilibrium.")\n'
    if marker in text and "Do not use its optimum as the record-board geometry" not in text:
        text = text.replace(marker, marker + warning, 1)

    opt.write_text(text)


if __name__ == "__main__":
    patch_app()
    patch_optimizer()
