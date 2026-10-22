"""
Simulateur de réseau d'antennes à balayage électronique (ULA / Phased Array)
Lancer :  streamlit run app.py
Dépendances : streamlit numpy scipy plotly pandas
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import windows as sw

st.set_page_config(page_title="Phased Array Simulator", page_icon="📡", layout="wide")

C0 = 299_792_458.0  # vitesse de la lumière (m/s)

WINDOWS = ["Uniforme (rectangulaire)", "Hamming", "Hann", "Blackman",
           "Kaiser", "Dolph-Chebyshev", "Taylor"]

DEFAULTS = dict(N=8, d=0.50, th0=0, win=WINDOWS[0], beta=5.0, att=30,
                bits=0, sig_a=0.0, sig_p=0.0, seed=1, q=0.0,
                freq=2.4, floor=-40, view="Cartésien", compare=False,
                dark=True)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ----------------------------------------------------------------------------
# Calcul
# ----------------------------------------------------------------------------
def amplitude_weights(name, N, beta=5.0, att=30):
    """Poids d'amplitude (tapering), normalisés à 1."""
    if N < 3 or name.startswith("Uniforme"):
        return np.ones(N)
    if name == "Hamming":
        w = sw.hamming(N)
    elif name == "Hann":            # fenêtre élargie pour que les bords ne soient pas nuls
        w = sw.hann(N + 2)[1:-1]
    elif name == "Blackman":
        w = sw.blackman(N + 2)[1:-1]
    elif name == "Kaiser":
        w = sw.kaiser(N, beta)
    elif name == "Dolph-Chebyshev":
        w = sw.chebwin(N, at=att)
    elif name == "Taylor":
        w = sw.taylor(N, nbar=4, sll=att)
    else:
        w = np.ones(N)
    return w / w.max()


def complex_weights(N, d, th0, w, bits=0, sig_a=0.0, sig_p=0.0, seed=1):
    """Poids complexes = amplitude * exp(j * phase progressive)."""
    n = np.arange(N)
    phase = -2 * np.pi * d * n * np.sin(np.radians(th0))   # Δφ = -k d sin(θ0)
    if bits > 0:                                           # déphaseurs à `bits` bits
        step = 2 * np.pi / 2 ** bits
        phase = np.round(phase / step) * step
    a = w.astype(float).copy()
    if sig_a > 0 or sig_p > 0:                             # erreurs aléatoires
        rng = np.random.default_rng(seed)
        a = a * (1 + sig_a / 100 * rng.standard_normal(N))
        phase = phase + np.radians(sig_p) * rng.standard_normal(N)
    return a * np.exp(1j * phase)


def pattern_db(theta, d, c, q=0.0):
    """Diagramme normalisé en dB (facteur de réseau x diagramme d'élément cos^q)."""
    n = np.arange(len(c))
    s = np.sin(np.radians(theta))
    af = np.exp(1j * 2 * np.pi * d * np.outer(s, n)) @ c
    elem = np.clip(np.cos(np.radians(theta)), 0, 1) ** q
    mag = np.abs(af) * elem
    return 20 * np.log10(mag / mag.max() + 1e-12)


def metrics(theta, dB, th0):
    """Retourne (HPBW en °, SLL en dB relatif au pic, indice du pic)."""
    pk = int(np.argmin(np.abs(theta - th0)))
    while pk > 0 and dB[pk - 1] > dB[pk]:
        pk -= 1
    while pk < len(dB) - 1 and dB[pk + 1] > dB[pk]:
        pk += 1
    peak = dB[pk]

    l = pk
    while l > 0 and dB[l - 1] <= dB[l]:
        l -= 1
    r = pk
    while r < len(dB) - 1 and dB[r + 1] <= dB[r]:
        r += 1
    side = np.concatenate([dB[:l], dB[r + 1:]])
    sll = side.max() - peak if side.size else np.nan

    def edge(step):
        i = pk
        while 0 <= i + step < len(dB) and dB[i + step] >= peak - 3:
            i += step
        j = i + step
        if not 0 <= j < len(dB):
            return theta[i]
        t = (dB[i] - (peak - 3)) / (dB[i] - dB[j])
        return theta[i] + t * (theta[j] - theta[i])

    return edge(1) - edge(-1), sll, pk


def array_gain_db(c, d, th0):
    """Gain du réseau vs un seul élément, dans la direction θ0."""
    n = np.arange(len(c))
    v = np.exp(1j * 2 * np.pi * d * n * np.sin(np.radians(th0)))
    return 10 * np.log10(np.abs(np.sum(c * v)) ** 2 / np.sum(np.abs(c) ** 2))


def quick(N, d, th0, name, beta, att, q=0.0, grid=1801):
    """Métriques idéales (sans erreurs) pour les études paramétriques."""
    th = np.linspace(-90, 90, grid)
    w = amplitude_weights(name, N, beta, att)
    c = complex_weights(N, d, th0, w)
    dB = pattern_db(th, d, c, q)
    h, s, _ = metrics(th, dB, th0)
    return h, s


# ----------------------------------------------------------------------------
# Barre latérale
# ----------------------------------------------------------------------------
sb = st.sidebar
sb.title("📡 Paramètres")
sb.button("↺ Réinitialiser", on_click=reset_all, use_container_width=True)

sb.subheader("Géométrie du réseau")
N = sb.slider("Nombre d'éléments (N)", 2, 64, key="N")
d = sb.slider("Espacement d/λ", 0.10, 2.00, step=0.01, key="d")
freq = sb.number_input("Fréquence (GHz) — pour convertir d en mm", 0.1, 100.0,
                       step=0.1, key="freq")

sb.subheader("Pointage")
th0 = sb.slider("Angle de pointage θ₀ (°)", -90, 90, key="th0")

sb.subheader("Pondération d'amplitude (tapering)")
win = sb.selectbox("Fenêtre", WINDOWS, key="win")
beta, att = st.session_state["beta"], st.session_state["att"]
if win == "Kaiser":
    beta = sb.slider("β (Kaiser)", 0.0, 12.0, step=0.5, key="beta")
elif win in ("Dolph-Chebyshev", "Taylor"):
    att = sb.slider("SLL cible (dB)", 20, 60, key="att")

with sb.expander("Imperfections matérielles (réalisme)"):
    bits = st.slider("Déphaseurs : bits (0 = idéal)", 0, 8, key="bits")
    sig_a = st.slider("Erreur d'amplitude σ (%)", 0.0, 30.0, step=0.5, key="sig_a")
    sig_p = st.slider("Erreur de phase σ (°)", 0.0, 30.0, step=0.5, key="sig_p")
    seed = st.number_input("Graine aléatoire", 0, 9999, key="seed")

with sb.expander("Diagramme d'élément"):
    q = st.slider("Exposant q de cos^q(θ)  (0 = isotrope, 1 ≈ patch)",
                  0.0, 3.0, step=0.1, key="q")

sb.subheader("Affichage")
view = sb.radio("Vue", ["Cartésien", "Polaire", "Les deux"], horizontal=True, key="view")
floor = sb.slider("Plancher (dB)", -80, -20, key="floor")
compare = sb.checkbox("Superposer le réseau uniforme", key="compare")
dark = sb.toggle("Graphiques sombres", key="dark")
template = "plotly_dark" if dark else "plotly_white"

# ----------------------------------------------------------------------------
# Calculs principaux
# ----------------------------------------------------------------------------
theta = np.linspace(-90, 90, 3601)
w = amplitude_weights(win, N, beta, att)
c = complex_weights(N, d, th0, w, bits, sig_a, sig_p, seed)
dB = pattern_db(theta, d, c, q)
hpbw, sll, pk = metrics(theta, dB, th0)
gain = array_gain_db(c, d, th0)
lam_mm = C0 / (freq * 1e9) * 1e3

dB_ref = None
if compare:
    c_ref = complex_weights(N, d, th0, np.ones(N))
    dB_ref = pattern_db(theta, d, c_ref, q)

# ----------------------------------------------------------------------------
# En-tête + métriques
# ----------------------------------------------------------------------------
st.title("Simulateur de réseau à balayage électronique")
st.caption("ULA — facteur de réseau, loi de phase progressive, tapering, quantification de phase")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Pointage θ₀", f"{th0:+.1f}°")
m2.metric("HPBW (−3 dB)", f"{hpbw:.1f}°")
m3.metric("SLL", f"{sll:.1f} dB" if np.isfinite(sll) else "—")
m4.metric("Gain du réseau", f"{gain:.1f} dB", help="Par rapport à un seul élément, dans la direction θ₀")
m5.metric("d = ", f"{d * lam_mm:.1f} mm", help=f"λ = {lam_mm:.1f} mm à {freq} GHz")

d_max = 1 / (1 + abs(np.sin(np.radians(th0))))
if d > d_max + 1e-9:
    st.warning(f"⚠️ d/λ = {d:.2f} > {d_max:.2f} : lobes de réseau (grating lobes) "
               f"visibles pour θ₀ = {th0}°. Condition : d/λ ≤ 1/(1+|sin θ₀|).")
else:
    st.success(f"✅ d/λ = {d:.2f} ≤ {d_max:.2f} : pas de lobe de réseau.")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Diagramme", "⚖️ Pondérations", "🔬 Étude paramétrique", "💾 Export & théorie"])

# ----------------------------------------------------------------------------
# Onglet 1 : diagramme
# ----------------------------------------------------------------------------
with tab1:
    y = np.maximum(dB, floor)

    def cartesian():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theta, y=y, name=win, fill="tozeroy",
                                 line=dict(width=2)))
        if dB_ref is not None:
            fig.add_trace(go.Scatter(x=theta, y=np.maximum(dB_ref, floor),
                                     name="Uniforme", line=dict(dash="dash", width=1.5)))
        fig.add_hline(y=-3, line_dash="dot", opacity=0.5, annotation_text="−3 dB")
        if np.isfinite(sll):
            fig.add_hline(y=dB[pk] + sll, line_dash="dot", opacity=0.5,
                          annotation_text=f"SLL {sll:.1f} dB")
        fig.add_vline(x=th0, line_dash="dot", opacity=0.5)
        fig.update_layout(template=template, height=480, xaxis_title="θ (°)",
                          yaxis_title="Niveau normalisé (dB)",
                          xaxis=dict(range=[-90, 90], dtick=15),
                          yaxis=dict(range=[floor, 0]),
                          legend=dict(orientation="h", y=1.08))
        return fig

    def polar():
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(theta=90 - theta, r=y, name=win, fill="toself"))
        if dB_ref is not None:
            fig.add_trace(go.Scatterpolar(theta=90 - theta, r=np.maximum(dB_ref, floor),
                                          name="Uniforme", line=dict(dash="dash")))
        fig.update_layout(template=template, height=480,
                          polar=dict(sector=[0, 180],
                                     radialaxis=dict(range=[floor, 0], angle=90),
                                     angularaxis=dict(tickvals=[0, 30, 60, 90, 120, 150, 180],
                                                      ticktext=["90°", "60°", "30°", "0°", "-30°", "-60°", "-90°"])),
                          legend=dict(orientation="h", y=1.08))
        return fig

    if view == "Cartésien":
        st.plotly_chart(cartesian(), use_container_width=True)
    elif view == "Polaire":
        st.plotly_chart(polar(), use_container_width=True)
    else:
        a, b = st.columns(2)
        a.plotly_chart(cartesian(), use_container_width=True)
        b.plotly_chart(polar(), use_container_width=True)

# ----------------------------------------------------------------------------
# Onglet 2 : pondérations
# ----------------------------------------------------------------------------
with tab2:
    phase_ideal = np.degrees(np.angle(np.exp(-1j * 2 * np.pi * d * np.arange(N) * np.sin(np.radians(th0)))))
    df_w = pd.DataFrame({
        "Élément n": np.arange(N),
        "Amplitude": np.abs(c),
        "Phase idéale (°)": phase_ideal,
        "Phase appliquée (°)": np.degrees(np.angle(c)),
    })
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Amplitude par élément", "Phase par élément (°)"))
    fig.add_trace(go.Bar(x=df_w["Élément n"], y=df_w["Amplitude"], name="Amplitude"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_w["Élément n"], y=df_w["Phase idéale (°)"], mode="lines+markers",
                             name="Idéale", line=dict(dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_w["Élément n"], y=df_w["Phase appliquée (°)"], mode="markers",
                             name="Appliquée", marker=dict(size=9, symbol="diamond")), row=1, col=2)
    fig.update_layout(template=template, height=400, legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_w.round(3), use_container_width=True, hide_index=True)
    if bits > 0:
        st.info(f"Pas de quantification : {360 / 2 ** bits:.2f}° ({bits} bits).")

# ----------------------------------------------------------------------------
# Onglet 3 : études paramétriques
# ----------------------------------------------------------------------------
with tab3:
    st.caption("Calculs idéaux (sans erreurs ni quantification), avec d/λ, fenêtre et q actuels.")

    st.subheader("1. Influence du nombre d'éléments N")
    Ns = np.arange(2, 65)
    res = np.array([quick(int(n), d, th0, win, beta, att, q) for n in Ns])
    f1 = make_subplots(rows=1, cols=2, subplot_titles=("HPBW (°)", "SLL (dB)"))
    f1.add_trace(go.Scatter(x=Ns, y=res[:, 0], name="HPBW simulée"), row=1, col=1)
    f1.add_trace(go.Scatter(x=Ns, y=np.degrees(0.886 / (Ns * d)), name="0,886·λ/(N·d)",
                            line=dict(dash="dot")), row=1, col=1)
    f1.add_trace(go.Scatter(x=Ns, y=res[:, 1], name="SLL"), row=1, col=2)
    f1.update_yaxes(range=[0, min(90, float(np.nanmax(res[:, 0])) * 1.1)], row=1, col=1)
    f1.update_xaxes(title_text="N")
    f1.update_layout(template=template, height=380, legend=dict(orientation="h", y=1.15))
    st.plotly_chart(f1, use_container_width=True)

    st.subheader("2. Comparaison des fenêtres (pour le N actuel)")
    rows = []
    for name in WINDOWS:
        h, s = quick(N, d, th0, name, 5.0, 30, q)
        rows.append({"Fenêtre": name, "HPBW (°)": round(h, 2), "SLL (dB)": round(s, 1)})
    df_win = pd.DataFrame(rows)
    f2 = go.Figure(go.Bar(x=df_win["Fenêtre"], y=df_win["SLL (dB)"]))
    f2.update_layout(template=template, height=340, yaxis_title="SLL (dB)")
    ca, cb = st.columns([3, 2])
    ca.plotly_chart(f2, use_container_width=True)
    cb.dataframe(df_win, use_container_width=True, hide_index=True)
    st.caption("Paramètres par défaut : Kaiser β = 5, Chebyshev/Taylor = −30 dB. "
               "Compromis : plus le SLL baisse, plus le lobe principal s'élargit.")

    st.subheader("3. Élargissement du faisceau en balayage")
    angles = np.arange(0, 81, 2)
    hp = np.array([quick(N, d, int(a), win, beta, att, 0.0)[0] for a in angles])
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=angles, y=hp, name="HPBW simulée"))
    f3.add_trace(go.Scatter(x=angles, y=hp[0] / np.cos(np.radians(angles)), name="HPBW(0)/cos θ₀",
                            line=dict(dash="dot")))
    f3.update_layout(template=template, height=340, xaxis_title="θ₀ (°)", yaxis_title="HPBW (°)")
    st.plotly_chart(f3, use_container_width=True)

# ----------------------------------------------------------------------------
# Onglet 4 : export & théorie
# ----------------------------------------------------------------------------
with tab4:
    e1, e2 = st.columns(2)
    df_pat = pd.DataFrame({"theta_deg": theta, "level_dB": dB})
    e1.download_button("⬇️ Diagramme (CSV)", df_pat.to_csv(index=False).encode(),
                       "pattern.csv", "text/csv", use_container_width=True)
    e2.download_button("⬇️ Pondérations complexes (CSV)",
                       pd.DataFrame({"n": np.arange(N), "amplitude": np.abs(c),
                                     "phase_deg": np.degrees(np.angle(c))}).to_csv(index=False).encode(),
                       "weights.csv", "text/csv", use_container_width=True)

    st.subheader("Rappels théoriques")
    st.latex(r"AF(\theta)=\sum_{n=0}^{N-1} w_n\,e^{\,j n\left[k d(\sin\theta-\sin\theta_0)\right]}")
    st.latex(r"\Delta\phi=-k\,d\,\sin\theta_0 \qquad k=\frac{2\pi}{\lambda}")
    st.latex(r"\mathrm{HPBW}\approx\frac{0{,}886\,\lambda}{N\,d\,\cos\theta_0}\quad(\text{uniforme})")
    st.latex(r"\frac{d}{\lambda}\le\frac{1}{1+|\sin\theta_0|}\ \Rightarrow\ \text{pas de grating lobe}")
    st.markdown("**Repères** : SLL uniforme ≈ −13,3 dB ; Hamming ≈ −42 dB ; "
                "Chebyshev/Taylor : SLL choisi (−20 à −60 dB).")
