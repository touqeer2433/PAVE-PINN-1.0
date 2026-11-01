"""PAVE-PINN Streamlit app: physics-constrained deep ensemble for ME life prediction of flexible pavements."""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pave_pinn_core import PavePINN

st.set_page_config(page_title="PAVE-PINN", page_icon="🛣️", layout="wide")

NAVY, OK, WARN, BAD = "#14284b", "#1a8f4c", "#d99a00", "#c62828"
st.markdown(f"""
<style>
.block-container {{padding-top: 3.2rem;}}
.hdr {{background:{NAVY}; color:white; padding:14px 22px; border-radius:10px; margin-bottom:14px;}}
.hdr h1 {{color:white; font-size:1.55rem; margin:0;}}
.hdr p {{color:#cfd8e8; margin:2px 0 0 0; font-size:0.92rem;}}
.card {{border:1px solid #dde3ec; border-radius:10px; padding:12px 16px; background:#fafbfd; height:100%;}}
.card h4 {{margin:0 0 6px 0; color:{NAVY}; font-size:1.0rem;}}
.big {{font-size:1.55rem; font-weight:700; color:#17202c;}}
.sub {{color:#5b6576; font-size:0.85rem;}}
.badge {{display:inline-block; padding:3px 10px; border-radius:12px; color:white; font-weight:700; font-size:0.8rem;}}
.decision {{border:2px solid; border-radius:10px; padding:10px 14px; font-weight:700; text-align:center; font-size:1.05rem;}}
</style>
<div class="hdr"><h1>PAVE-PINN</h1>
<p>Physics-constrained deep ensemble for mechanistic-empirical fatigue and rutting life of six-layer flexible pavements,
trained on 3,492 viscoelastic-elastoplastic ABAQUS simulations</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return PavePINN()


M = load_model()
meta = M.meta
FI = {f: i for i, f in enumerate(M.features)}
LAYERS = meta["layers"]
LNAME = dict(zip(LAYERS, meta["layer_names"]))

LABEL = {"t": ("Thickness", "mm"), "E": ("Elastic modulus", "MPa"), "ν": ("Poisson's ratio", "-"), "ρ": ("Density", "t/mm³"),
         "φ": ("Friction angle", "°"), "K": ("Flow stress ratio K", "-"), "ψ": ("Dilation angle", "°"), "σy": ("Yield stress", "MPa")}


def lab(f):
    if f == "P": return "Tyre pressure P (MPa)"
    if f == "T": return "Temperature T (°C)"
    if f == "S": return "Saturation S (-)"
    sym, layer = f.split("_")
    name, unit = LABEL[sym]
    return f"{name}, {layer} ({unit})"


def num(f, key=None, step=None, fmt=None):
    i = FI[f]
    lo, hi, d = M.x_min[i], M.x_max[i], M.x_default[i]
    if fmt is None:
        fmt = "%.3e" if d < 1e-6 else ("%.4f" if d < 1 else ("%.2f" if d < 100 else "%.1f"))
    return st.number_input(lab(f), value=float(d), min_value=0.0, step=step or float((hi - lo) / 50) or None,
                           format=fmt, key=key or f, help=f"Training range: {lo:.4g} to {hi:.4g}")


# ------------------------------------------------------------------ inputs
with st.sidebar:
    st.header("Design inputs")
    n_esal = st.number_input("Design traffic (ESALs)", value=float(meta["design_esals"]), min_value=1.0, step=1e5, format="%.0f")
    st.subheader("Layer thickness (mm)")
    for k in LAYERS:
        num(f"t_{k}")
    st.subheader("Elastic modulus (MPa)")
    for k in LAYERS:
        num(f"E_{k}")
    st.subheader("Load and environment")
    num("P", step=0.01); num("T", step=0.5)
    with st.expander("Advanced: Poisson's ratio, density, Drucker-Prager, saturation"):
        for k in LAYERS:
            num(f"ν_{k}", step=0.01)
        for k in LAYERS:
            num(f"ρ_{k}", step=1e-10)
        for k in ["AggB", "SB", "Fill", "SG"]:
            for s in ["φ", "K", "ψ", "σy"]:
                num(f"{s}_{k}")
        num("S", step=0.05)
    st.caption("Values outside the training range are flagged; predictions there are extrapolations.")

x = np.array([st.session_state.get(f, M.x_default[i]) for i, f in enumerate(M.features)], float)
res = M.predict(pd.DataFrame([x], columns=M.features), n_esal, fields=True).iloc[0]
Y, _, _ = M.raw(x[None]); Y = Y[0]
outside = [f for f in res["inputs_outside_training_range"].split(", ") if f]


def risk(D):
    lo, hi = meta["risk"]
    if D >= hi: return "HIGH RISK", BAD
    if D >= lo: return "MODERATE RISK", WARN
    return "LOW RISK", OK


sci = lambda v: f"{v:.2e}".replace("e+0", "E+").replace("e+", "E+")

tab1, tab2, tab3, tab4 = st.tabs(["Design check", "Layer responses", "Batch prediction", "About the model"])

# ------------------------------------------------------------------ tab 1
with tab1:
    if outside:
        st.warning("Outside the training range: " + ", ".join(outside) + ". This prediction is an extrapolation.")
    c = st.columns(4)
    cards = [("Tensile strain εt", f"{res.et_microstrain:.1f} µε", f"95%: {res.et_low:.1f} to {res.et_high:.1f}", "bottom of asphalt base course"),
             ("Compressive strain εc", f"{res.ec_microstrain:.1f} µε", f"95%: {res.ec_low:.1f} to {res.ec_high:.1f}", "top of subgrade fill"),
             ("Fatigue life Nf", sci(res.Nf), f"95%: {sci(res.Nf_low)} to {sci(res.Nf_high)}", "Asphalt Institute (1982)"),
             ("Rutting life Nr", sci(res.Nr), f"95%: {sci(res.Nr_low)} to {sci(res.Nr_high)}", "Asphalt Institute (1982)")]
    for col, (t, v, iv, note) in zip(c, cards):
        col.markdown(f"<div class='card'><h4>{t}</h4><div class='big'>{v}</div><div class='sub'>{iv}</div><div class='sub'>{note}</div></div>", unsafe_allow_html=True)

    st.write("")
    left, mid, right = st.columns([1.05, 1.0, 1.15])
    with left.container(border=True):
        st.markdown("<h4 style='margin:0;color:#14284b'>Distress and damage</h4>", unsafe_allow_html=True)
        for name, D in [("Fatigue cracking", res.Df), ("Subgrade rutting", res.Dr)]:
            t, col = risk(D)
            st.markdown(f"**{name}**: damage ratio D = {D:.3f} &nbsp; <span class='badge' style='background:{col}'>{t}</span>"
                        f"<br><span class='sub'>remaining life {max(0.0, (1 - D) * 100):.1f}% at {n_esal:,.0f} ESALs</span>", unsafe_allow_html=True)
        mode = "Rutting" if res.Dr > res.Df else "Fatigue"
        st.markdown(f"**Governing mode:** {mode} &nbsp; | &nbsp; **Governing life:** {sci(res.design_life)} ESALs", unsafe_allow_html=True)
        D = res.D
        txt, col = (("DESIGN INADEQUATE: STRENGTHEN SECTION", BAD) if D >= meta["risk"][1] else
                    ("DESIGN MARGINAL: REVIEW SECTION", WARN) if D >= meta["risk"][0] else
                    ("DESIGN ADEQUATE FOR THE DESIGN TRAFFIC", OK))
        st.markdown(f"<div class='decision' style='color:{col}; border-color:{col}'>{txt}</div>", unsafe_allow_html=True)

    with mid.container(border=True):
        st.markdown("<h4 style='margin:0;color:#14284b'>Pavement section</h4>", unsafe_allow_html=True)
        cols = ["#2b2b2b", "#555555", "#b9a98a", "#d6c7a1", "#c79a6b", "#a9794d"]
        t = [x[FI[f"t_{k}"]] for k in LAYERS]
        fig = go.Figure()
        depth = 0.0
        for k, th, cl in zip(LAYERS, t, cols):
            fig.add_trace(go.Bar(x=[1], y=[th], base=[depth], marker_color=cl, width=0.6, name=k,
                                 hovertemplate=f"{LNAME[k]}: {th:.0f} mm<extra></extra>", showlegend=False))
            fig.add_annotation(x=1.36, y=depth + th / 2, text=f"{k}  {th:.0f} mm", showarrow=False, xanchor="left", font=dict(size=11))
            depth += th
        z_et = t[0] + t[1]; z_ec = sum(t[:4])
        fig.add_trace(go.Scatter(x=[1, 1], y=[z_et, z_ec], mode="markers+text", marker=dict(size=11, color="#ffd54a", line=dict(color="black", width=1)),
                                 text=["εt", "εc"], textposition="middle left", textfont=dict(color=BAD, size=13), showlegend=False, hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed", title="Depth (mm)")
        fig.update_xaxes(visible=False, range=[0.4, 2.0])
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", barmode="overlay")
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"<span class='sub'>Tyre pressure {x[FI['P']]:.2f} MPa on a 150 mm radius</span>", unsafe_allow_html=True)

    with right.container(border=True):
        st.markdown("<h4 style='margin:0;color:#14284b'>Reliability and design advice</h4>", unsafe_allow_html=True)
        n_in = len(M.features) - len(outside)
        st.markdown(f"Inputs inside training envelope: **{n_in} / {len(M.features)}**")
        tot, fails = M.physics_checks(x)
        colr = OK if not fails else WARN
        st.markdown(f"Physics checks passed (this design): <b style='color:{colr}'>{tot - len(fails)} / {tot}</b>", unsafe_allow_html=True)
        st.caption("Not satisfied: " + "; ".join(fails) if fails else "Sensitivity signs, surface traction and stress attenuation all satisfied.")
        cand = [FI[f"t_{k}"] for k in LAYERS[:5]] + [FI[f"E_{k}"] for k in LAYERS]
        S = M.sensitivities(x, cand)
        top = sorted(((M.features[j], S[j][3]) for j in cand), key=lambda a: -a[1])[:3]
        st.markdown(f"**Most effective changes for {mode.lower()} life** (% gain in governing life per 1% increase):")
        for f, g in top:
            st.markdown(f"- {lab(f)}: **{g:+.2f}%**")

    st.write("")
    single = pd.DataFrame([dict(zip(M.features, x)) | {"n_ESALs": n_esal} | res.drop([c for c in res.index if "_" in c and c.split("_")[-1] in LAYERS]).to_dict()])
    st.download_button("Download this design and its predictions (CSV)", single.to_csv(index=False).encode(), "PAVE_PINN_design.csv", "text/csv")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown("Layer-wise extreme responses, as exported from ABAQUS and predicted by the ensemble (6 layers × 11 channels).")
    tbl = pd.DataFrame(Y, index=[f"{k} ({LNAME[k]})" for k in LAYERS],
                       columns=[f"{c} ({u})" for c, u in zip(meta["channels"], meta["channel_units"])])
    st.dataframe(tbl.style.format("{:.4g}"), width="stretch")
    iS, iU = meta["channels"].index("S22"), meta["channels"].index("U2")
    zt = np.cumsum([0] + [x[FI[f"t_{k}"]] for k in LAYERS])[:-1]
    c1, c2 = st.columns(2)
    for col, ch, idx, unit, scale in [(c1, "Vertical stress |S22|", iS, "kPa", 1000), (c2, "Vertical deflection |U2|", iU, "mm", 1)]:
        f2 = go.Figure(go.Scatter(x=np.abs(Y[:, idx]) * scale, y=zt, mode="lines+markers", line=dict(color=NAVY), text=LAYERS,
                                  hovertemplate="%{text}: %{x:.3g}<extra></extra>"))
        f2.update_yaxes(autorange="reversed", title="Depth to top of layer (mm)")
        f2.update_xaxes(title=f"{ch} ({unit})")
        f2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title=ch, plot_bgcolor="white")
        col.plotly_chart(f2, width="stretch")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("Upload a CSV or Excel file with one design per row. Columns use the parameter names below; "
                "missing columns take the training median.")
    tmpl = pd.DataFrame([M.x_default], columns=M.features)
    st.download_button("Download input template (CSV)", tmpl.to_csv(index=False).encode("utf-8-sig"), "PAVE_PINN_template.csv", "text/csv")
    up = st.file_uploader("Designs file", type=["csv", "xlsx", "xls"])
    want_fields = st.checkbox("Include all 66 layer responses in the output", value=False)
    if up is not None:
        try:
            df = pd.read_csv(up, encoding="utf-8-sig") if up.name.lower().endswith(".csv") else pd.read_excel(up)
            recog = [c for c in df.columns if str(c).strip() in FI]
            out = M.predict(df, n_esal, fields=want_fields)
            full = pd.concat([df.reset_index(drop=True), out], axis=1)
            st.success(f"{len(df)} designs predicted; {len(recog)} of {len(M.features)} input columns recognised.")
            st.dataframe(full.head(200), width="stretch")
            buf = io.BytesIO(); full.to_csv(buf, index=False)
            st.download_button("Download predictions (CSV)", buf.getvalue(), "PAVE_PINN_batch_predictions.csv", "text/csv")
        except Exception as e:  # pragma: no cover
            st.error(f"Could not read the file: {e}")

# ------------------------------------------------------------------ tab 4
with tab4:
    r2 = meta["test_r2"]
    st.markdown(f"""
**PAVE-PINN** maps 43 design parameters of a six-layer flexible pavement (asphalt wearing course, asphalt base, aggregate base,
granular subbase, subgrade fill, natural subgrade) to the layer-wise extreme responses and to the fatigue and rutting lives.

* **Training data:** 3,492 viscoelastic (asphalt) – Drucker-Prager (unbound) ABAQUS simulations; 2,444 used for training.
* **Network:** residual MLP with hard sign constraints in the output layer; physics loss on the surface traction, the attenuation of stress
  and deflection with depth and 29 audited sensitivity signs; five-seed deep ensemble with calibrated 95% intervals.
* **Critical strains:** {meta['definitions']['et']}; {meta['definitions']['ec']}.
* **Transfer functions (Asphalt Institute, 1982):** Nf = 0.0796 εt^-3.291 E_ACB(psi)^-0.854; Nr = 1.365×10⁻⁹ εc^-4.477; damage D = n/N (Miner).
* **Test accuracy (525 held-out designs):** R² εt {r2['εt (µε)']:.3f}, εc {r2['εc (µε)']:.3f}, log Nf {r2['log10 Nf']:.3f}, log Nr {r2['log10 Nr']:.3f}.

**Limitations.** Valid inside the training ranges only. Static circular load (150 mm radius), fully bonded layers, no moving or
non-uniform contact stresses. Absolute lives inherit the Asphalt Institute calibration and should be recalibrated locally before design use.
The model is validated against the FE model, not against field measurements.

**Citation.** Rind, T. A., Ahmed, S., Waqas, H. A. PAVE-PINN: a physics-constrained deep ensemble for mechanistic-empirical life prediction
of flexible pavements from 3,500 viscoelastic-elastoplastic finite element simulations (under review).
""")
    rng = pd.DataFrame({"parameter": M.features, "label": [lab(f) for f in M.features], "min": M.x_min, "max": M.x_max, "default (median)": M.x_default})
    with st.expander("Training ranges of the 43 parameters"):
        st.dataframe(rng, width="stretch", hide_index=True)
