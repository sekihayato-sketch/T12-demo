import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import hashlib
from math import exp, factorial, log2, sqrt, ceil

try:
    from scipy.stats import beta
    from scipy.optimize import linprog
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="T12 2013 論文再現シミュレータ", layout="wide")

# ============================================================
# Paper: Lucamarini et al., Optics Express 21, 24550 (2013)
# Efficient decoy-state quantum key distribution with quantified security
# ============================================================
PULSE_RATE_HZ = 1_000_000_000
SESSION_SECONDS = 20 * 60
N_SESSION = PULSE_RATE_HZ * SESSION_SECONDS

MU_U, MU_V, MU_W = 0.425, 0.044, 0.001
MUS = {"u": MU_U, "v": MU_V, "w": MU_W}
P_W = 1 / 256
P_V = 1 / 128
P_U = 1 - P_V - P_W
P_INTENSITY = {"u": P_U, "v": P_V, "w": P_W}

EPS_TOTAL = 1e-10
DEFAULT_EPS_PE = 1e-15
DEFAULT_EPS_S = 1e-12
DEFAULT_EPS_EC = 1e-12
DEFAULT_F_EC = 1.10
DEFAULT_KMAX = 12

# Experimental T12 50 km values reported for a 20 min session.
# C: sifted/non-empty counts after same-basis selection.
# QBER is reported only for signal u in Z and X.
PAPER_T12_50KM_COUNTS = {
    ("u", "Z"): 5.016e9,
    ("u", "X"): 2.231e7,
    ("v", "Z"): 6.21e6,
    ("v", "X"): 2.843e4,
    ("w", "Z"): 1.259e6,
    ("w", "X"): 5.79e3,
}
PAPER_T12_50KM_QBER = {"Z": 0.0426, "X": 0.0364}

BENCHMARKS = pd.DataFrame([
    {"距離[km]": 35, "T12 SKR[Mb/s]": 2.20, "BB84 SKR[Mb/s]": 1.18},
    {"距離[km]": 50, "T12 SKR[Mb/s]": 1.09, "BB84 SKR[Mb/s]": 0.63},
    {"距離[km]": 65, "T12 SKR[Mb/s]": 0.40, "BB84 SKR[Mb/s]": 0.26},
    {"距離[km]": 80, "T12 SKR[Mb/s]": 0.12, "BB84 SKR[Mb/s]": 0.06},
])


def h2(x: float) -> float:
    x = min(max(float(x), 1e-15), 1 - 1e-15)
    return -x * log2(x) - (1 - x) * log2(1 - x)


def poisson_weights(mu: float, kmax: int) -> np.ndarray:
    return np.array([exp(-mu) * mu**k / factorial(k) for k in range(kmax + 1)], dtype=float)


def fmt_si(x, suffix="", digits=3):
    try:
        x = float(x)
    except Exception:
        return str(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    for scale, unit in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")]:
        if x >= scale:
            return f"{sign}{x/scale:.{digits}f}{unit}{suffix}"
    return f"{sign}{x:.{digits}f}{suffix}"


def cp_interval(success: int, total: int, alpha: float):
    if total <= 0:
        return 0.0, 1.0
    success = int(max(0, min(success, total)))
    if not SCIPY_OK:
        # Conservative fallback: Wilson-like normal interval. Use SciPy for paper-faithful CP.
        p = success / total
        rad = sqrt(max(p * (1 - p), 1e-18) * log(2 / max(alpha, 1e-300)) / total)
        return max(0.0, p - rad), min(1.0, p + rad)
    a = max(alpha / 2, 1e-300)
    lo = 0.0 if success == 0 else beta.ppf(a, success, total - success + 1)
    hi = 1.0 if success == total else beta.ppf(1 - a, success + 1, total - success)
    return float(lo), float(hi)


def delta_security(n_raw: float, eps_total: float, eps_s: float, eps_pe: float, eps_ec: float):
    # Eq.(3) OCR in the paper is difficult to read in plain text.
    # This implements the commonly used form shown there:
    # Delta = 7 sqrt(n log2(2/(eps_s-eps_pe))) + 2 log2(1/(2*(eps_total-eps_ec-eps_s))).
    n_raw = max(float(n_raw), 1.0)
    t1 = max(eps_s - eps_pe, 1e-300)
    t2 = max(eps_total - eps_ec - eps_s, 1e-300)
    return 7.0 * sqrt(n_raw * log2(2.0 / t1)) + 2.0 * log2(1.0 / (2.0 * t2))


def make_counts_from_paper(p_x: float, qber_z: float, qber_x: float, randomize=False, seed=1):
    """Use the paper's 50 km measured T12 yields to synthesize counts for any basis bias.

    For T12 pX=1/16 this reproduces the reported C_jZZ/C_jXX values. For BB84 pX=1/2,
    it keeps the same intensity/basis yields and changes only basis probabilities.
    """
    rng = np.random.default_rng(seed)
    p_z = 1 - p_x
    counts = {"N_total": int(N_SESSION), "p_x": p_x}
    # Estimate per-intensity/per-basis yields from the paper's T12 50 km session.
    p_z_t12 = 15 / 16
    p_x_t12 = 1 / 16
    for lab in ["u", "v", "w"]:
        for basis, p_basis, p_basis_t12 in [("Z", p_z, p_z_t12), ("X", p_x, p_x_t12)]:
            N = N_SESSION * P_INTENSITY[lab] * p_basis * p_basis
            paper_N = N_SESSION * P_INTENSITY[lab] * p_basis_t12 * p_basis_t12
            Y = PAPER_T12_50KM_COUNTS[(lab, basis)] / paper_N
            mean_C = N * Y
            C = rng.poisson(mean_C) if randomize else int(round(mean_C))
            if lab == "u":
                qber = qber_z if basis == "Z" else qber_x
                E = rng.binomial(C, qber) if randomize else int(round(C * qber))
            else:
                # v,w errors are not used by Eq.(7)/(23), but keep approximate values for display.
                qber = qber_z if basis == "Z" else qber_x
                E = int(round(C * qber))
            counts[("N", lab, basis)] = int(round(N))
            counts[("C", lab, basis)] = int(C)
            counts[("E", lab, basis)] = int(E)
    return counts


def make_counts_from_channel(p_x: float, distance_km: float, alpha_db_per_km: float, spd_eff: float,
                             rx_loss_db: float, dark_per_gate: float, eopt_z: float, eopt_x: float,
                             session_seconds: int, randomize=False, seed=1):
    rng = np.random.default_rng(seed)
    p_z = 1 - p_x
    N_total = int(PULSE_RATE_HZ * session_seconds)
    eta = spd_eff * 10 ** (-(alpha_db_per_km * distance_km + rx_loss_db) / 10)
    counts = {"N_total": N_total, "p_x": p_x, "eta": eta}
    for lab, mu in MUS.items():
        # threshold click probability for a phase-randomized coherent pulse
        photon_click = 1 - exp(-eta * mu)
        click_prob = 1 - (1 - photon_click) * (1 - dark_per_gate)
        for basis, p_basis, eopt in [("Z", p_z, eopt_z), ("X", p_x, eopt_x)]:
            N = N_total * P_INTENSITY[lab] * p_basis * p_basis
            C_mean = N * click_prob
            C = rng.poisson(C_mean) if randomize else int(round(C_mean))
            dark_only_frac = dark_per_gate / max(click_prob, 1e-30)
            qber = min(0.5, eopt * (1 - dark_only_frac) + 0.5 * dark_only_frac)
            E = rng.binomial(C, qber) if randomize else int(round(C * qber))
            counts[("N", lab, basis)] = int(round(N))
            counts[("C", lab, basis)] = int(C)
            counts[("E", lab, basis)] = int(E)
    return counts


def cp_bounds_for_basis(counts, basis: str, eps_pe: float, kmax: int):
    # Use a simple union split over 3 detection-rate bounds + 1 signal-error bound per basis.
    alpha_y = eps_pe / 12.0
    alpha_b = eps_pe / 4.0
    Y_bounds = {}
    for lab in ["u", "v", "w"]:
        N = counts[("N", lab, basis)]
        C = counts[("C", lab, basis)]
        Y_bounds[lab] = cp_interval(C, N, alpha_y)
    # B is the bit-error rate per emitted pulse in Eq.(10)/(12), bounded from E out of N.
    N_u = counts[("N", "u", basis)]
    E_u = counts[("E", "u", basis)]
    B_u = cp_interval(E_u, N_u, alpha_b)
    return Y_bounds, B_u


def minimize_y(counts, basis: str, target_k: int, eps_pe: float, kmax: int):
    Y_bounds, _ = cp_bounds_for_basis(counts, basis, eps_pe, kmax)
    A_ub, b_ub = [], []
    for lab, mu in MUS.items():
        w = poisson_weights(mu, kmax)
        lo, hi = Y_bounds[lab]
        A_ub.append(w)
        b_ub.append(hi)
        A_ub.append(-w)
        b_ub.append(-lo)
    c = np.zeros(kmax + 1)
    c[target_k] = 1.0
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, 1)] * (kmax + 1), method="highs")
    if not res.success:
        return 0.0, {"lp_success": False, "message": res.message, "Y_bounds": Y_bounds}
    return float(max(0.0, res.x[target_k])), {"lp_success": True, "Y_bounds": Y_bounds, "x": res.x}


def secure_bits_for_basis(counts, key_basis: str, phase_basis: str, eps_total: float, eps_s: float,
                          eps_pe: float, eps_ec: float, f_ec: float, kmax: int):
    y0_key, meta_y0 = minimize_y(counts, key_basis, 0, eps_pe, kmax)
    y1_key, meta_y1 = minimize_y(counts, key_basis, 1, eps_pe, kmax)
    y0_phase, _ = minimize_y(counts, phase_basis, 0, eps_pe, kmax)
    y1_phase, _ = minimize_y(counts, phase_basis, 1, eps_pe, kmax)
    _, B_u_phase = cp_bounds_for_basis(counts, phase_basis, eps_pe, kmax)
    B_u_phase_plus = B_u_phase[1]

    denom = exp(-MU_U) * MU_U * max(y1_phase, 1e-300)
    q1_phase = (B_u_phase_plus - 0.5 * exp(-MU_U) * y0_phase) / denom
    q1_phase = min(max(q1_phase, 0.0), 0.5)

    N_u = counts[("N", "u", key_basis)]
    C_u = counts[("C", "u", key_basis)]
    E_u = counts[("E", "u", key_basis)]
    Q_u = E_u / max(C_u, 1)

    S0 = N_u * exp(-MU_U) * y0_key
    S1 = N_u * exp(-MU_U) * MU_U * y1_key
    leak_ec = C_u * f_ec * h2(Q_u)
    Delta = delta_security(C_u, eps_total, eps_s, eps_pe, eps_ec)
    L = S0 + S1 * (1 - h2(q1_phase)) - leak_ec - Delta
    L = int(max(0, np.floor(L)))
    rate_mbps = PULSE_RATE_HZ * L / max(counts["N_total"], 1) / 1e6
    return {
        "basis": key_basis,
        "N_u": N_u,
        "C_u": C_u,
        "E_u": E_u,
        "QBER[%]": 100 * Q_u,
        "y0_min": y0_key,
        "y1_min": y1_key,
        "phase_basis": phase_basis,
        "q1_phase_max[%]": 100 * q1_phase,
        "S0": S0,
        "S1": S1,
        "leakEC": leak_ec,
        "Delta": Delta,
        "secure_bits": L,
        "SKR[Mb/s]": rate_mbps,
        "lp_ok": meta_y0.get("lp_success", False) and meta_y1.get("lp_success", False),
    }


def finalize(counts, protocol_name: str, eps_total: float, eps_s: float, eps_pe: float, eps_ec: float, f_ec: float, kmax: int):
    z = secure_bits_for_basis(counts, "Z", "X", eps_total, eps_s, eps_pe, eps_ec, f_ec, kmax)
    x = secure_bits_for_basis(counts, "X", "Z", eps_total, eps_s, eps_pe, eps_ec, f_ec, kmax)
    final_bits = z["secure_bits"] + x["secure_bits"]
    rate = PULSE_RATE_HZ * final_bits / max(counts["N_total"], 1) / 1e6
    sifted = sum(counts[("C", "u", b)] for b in ["Z", "X"])
    return {"Protocol": protocol_name, "N_total": counts["N_total"], "sifted_u": sifted,
            "final_bits": final_bits, "SKR[Mb/s]": rate, "Z": z, "X": x, "counts": counts}


def key_preview(nbits: int):
    if nbits <= 0:
        return "-"
    seed = f"t12-paper-preview-{nbits}".encode()
    out = ""
    ctr = 0
    while len(out) < min(nbits, 4096):
        out += "".join(f"{b:08b}" for b in hashlib.sha256(seed + str(ctr).encode()).digest())
        ctr += 1
    return out[: min(nbits, 4096)]


def result_tables(results):
    overall = []
    basis_rows = []
    for r in results:
        overall.append({
            "プロトコル": r["Protocol"],
            "送信パルス数": r["N_total"],
            "信号u sifted counts": r["sifted_u"],
            "最終鍵長[bit]": r["final_bits"],
            "SKR[Mb/s]": r["SKR[Mb/s]"],
        })
        for b in ["Z", "X"]:
            s = r[b]
            basis_rows.append({
                "プロトコル": r["Protocol"],
                "基底": b,
                "C_u": s["C_u"],
                "QBER[%]": s["QBER[%]"],
                "y0_min": s["y0_min"],
                "y1_min": s["y1_min"],
                "q1_phase_max[%]": s["q1_phase_max[%]"],
                "S0[bit]": s["S0"],
                "S1[bit]": s["S1"],
                "leakEC[bit]": s["leakEC"],
                "Delta[bit]": s["Delta"],
                "安全鍵長[bit]": s["secure_bits"],
                "SKR[Mb/s]": s["SKR[Mb/s]"],
            })
    return pd.DataFrame(overall), pd.DataFrame(basis_rows)


st.title("T12 2013 論文再現シミュレータ")
st.caption("CP信頼区間 + 線形計画法 + Eq.(7)/(23) に寄せた、論文再現用の教育・検証アプリです。")

st.markdown("""
この版では、以前の `固定PA圧縮率 0.292` や単純な2-decoy近似ではなく、論文の流れに近づけています。

1. 実験で得る量：`N_{μDD}`, `C_{μDD}`, `E_{μDD}` を基底・強度別に作る  
2. 統計：Clopper-Pearson信頼区間で `Y^-`, `Y^+`, `B^-`, `B^+` を作る  
3. 推定：線形計画法で `y0`, `y1` を最小化し、Eq.(23)で位相誤り `q1` を最悪化する  
4. 鍵率：Eq.(7)に近い形で、Z基底とX基底の安全鍵長を別々に計算して合算する

注意：これは「secure key length / PA圧縮後の鍵長」を再現する計算器です。LDPCの実復号器、Toeplitz hashingの実装、APDアフターパルスの時系列相関までは実装していません。
""")

if not SCIPY_OK:
    st.error("このアプリには scipy が必要です。実行環境で `pip install scipy` を実施してください。")
    st.stop()

with st.sidebar:
    st.header("再現モード")
    mode = st.radio("入力データ", ["論文50 km実測カウント", "物理チャネルモデル"], index=0)
    randomize = st.checkbox("統計揺らぎをサンプルする", value=False)
    seed = st.number_input("乱数seed", min_value=0, value=1, step=1)

    st.header("プロトコル")
    show_bb84 = st.checkbox("BB84参照も計算", value=True)

    st.header("セキュリティ・後処理")
    eps_total = st.number_input("epsilon total", value=EPS_TOTAL, format="%.1e")
    eps_pe = st.number_input("epsilon PE", value=DEFAULT_EPS_PE, format="%.1e")
    eps_s = st.number_input("epsilon smoothing", value=DEFAULT_EPS_S, format="%.1e")
    eps_ec = st.number_input("epsilon EC/verification", value=DEFAULT_EPS_EC, format="%.1e")
    f_ec = st.slider("EC効率 fEC", 1.0, 1.5, DEFAULT_F_EC, 0.01)
    kmax = st.slider("Poisson打切り kmax", 5, 30, DEFAULT_KMAX, 1)

    if mode == "論文50 km実測カウント":
        st.header("論文50 km QBER")
        qber_z = st.slider("Q_Z,u [%]", 0.0, 10.0, 4.26, 0.01) / 100
        qber_x = st.slider("Q_X,u [%]", 0.0, 10.0, 3.64, 0.01) / 100
    else:
        st.header("物理チャネル")
        distance_km = st.slider("距離 [km]", 0.0, 100.0, 50.0, 1.0)
        alpha = st.slider("ファイバ損失 [dB/km]", 0.15, 0.30, 0.20, 0.01)
        spd_eff = st.slider("APD/SPD効率 [%]", 1.0, 50.0, 20.5, 0.1) / 100
        rx_loss_db = st.slider("受信光学損失 [dB]", 0.0, 10.0, 0.0, 0.1)
        dark = st.number_input("暗計数確率 / gate", value=2.1e-5, format="%.2e")
        eopt_z = st.slider("Z光学誤差 [%]", 0.0, 10.0, 4.26, 0.01) / 100
        eopt_x = st.slider("X光学誤差 [%]", 0.0, 10.0, 3.64, 0.01) / 100
        session_seconds = st.select_slider("セッション時間 [s]", options=[1, 10, 60, 600, 1200], value=1200)

if st.button("論文式で計算", type="primary"):
    if eps_s <= eps_pe:
        st.error("Eq.(3)のため、epsilon smoothing は epsilon PE より大きくしてください。")
        st.stop()
    if eps_total <= eps_ec + eps_s:
        st.error("Eq.(3)のため、epsilon total は epsilon EC + epsilon smoothing より大きくしてください。")
        st.stop()

    results = []
    if mode == "論文50 km実測カウント":
        t12_counts = make_counts_from_paper(1 / 16, qber_z, qber_x, randomize, seed)
        results.append(finalize(t12_counts, "T12 pX=1/16", eps_total, eps_s, eps_pe, eps_ec, f_ec, kmax))
        if show_bb84:
            bb_counts = make_counts_from_paper(1 / 2, qber_z, qber_x, randomize, seed + 100)
            results.append(finalize(bb_counts, "BB84 pX=1/2", eps_total, eps_s, eps_pe, eps_ec, f_ec, kmax))
    else:
        t12_counts = make_counts_from_channel(1 / 16, distance_km, alpha, spd_eff, rx_loss_db, dark, eopt_z, eopt_x, session_seconds, randomize, seed)
        results.append(finalize(t12_counts, "T12 pX=1/16", eps_total, eps_s, eps_pe, eps_ec, f_ec, kmax))
        if show_bb84:
            bb_counts = make_counts_from_channel(1 / 2, distance_km, alpha, spd_eff, rx_loss_db, dark, eopt_z, eopt_x, session_seconds, randomize, seed + 100)
            results.append(finalize(bb_counts, "BB84 pX=1/2", eps_total, eps_s, eps_pe, eps_ec, f_ec, kmax))

    overall_df, basis_df = result_tables(results)
    st.subheader("1. 全体結果")
    st.dataframe(overall_df, use_container_width=True)
    fig = px.bar(overall_df, x="プロトコル", y="SKR[Mb/s]", text="SKR[Mb/s]", title="Secure key rate")
    fig.update_traces(texttemplate="%{y:.3f} Mb/s", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. 基底別の鍵率分解")
    st.dataframe(basis_df, use_container_width=True)
    fig2 = px.bar(basis_df, x="プロトコル", y="SKR[Mb/s]", color="基底", barmode="group", text="SKR[Mb/s]", title="Basis-dependent secure key rate")
    fig2.update_traces(texttemplate="%{y:.3f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. 論文ベンチマーク")
    st.dataframe(BENCHMARKS, use_container_width=True)
    if mode == "論文50 km実測カウント":
        bench50 = BENCHMARKS[BENCHMARKS["距離[km]"] == 50].iloc[0]
        rows = []
        for r in results:
            target = bench50["T12 SKR[Mb/s]"] if "T12" in r["Protocol"] else bench50["BB84 SKR[Mb/s]"]
            rows.append({"プロトコル": r["Protocol"], "今回SKR[Mb/s]": r["SKR[Mb/s]"], "論文50km[Mb/s]": target, "差分[Mb/s]": r["SKR[Mb/s]"] - target})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("4. 実験統計 N/C/E")
    show_rows = []
    for r in results:
        c = r["counts"]
        for lab in ["u", "v", "w"]:
            for b in ["Z", "X"]:
                show_rows.append({
                    "プロトコル": r["Protocol"], "強度": lab, "基底": b,
                    "N": c[("N", lab, b)], "C": c[("C", lab, b)], "E": c[("E", lab, b)],
                    "Y=C/N": c[("C", lab, b)] / max(c[("N", lab, b)], 1),
                    "B=E/N": c[("E", lab, b)] / max(c[("N", lab, b)], 1),
                })
    st.dataframe(pd.DataFrame(show_rows), use_container_width=True)

    st.subheader("5. 最終鍵プレビュー")
    for r in results:
        with st.expander(r["Protocol"]):
            st.caption("表示はSHA-256から作ったデモ用プレビューです。実際のPA後の鍵ではありません。")
            st.code(key_preview(r["final_bits"]), language="text")
else:
    st.info("左の条件を確認して、［論文式で計算］を押してください。")
