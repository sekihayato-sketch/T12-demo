import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import time
from math import exp, factorial, log2, sqrt, floor
from scipy.stats import norm
from scipy.optimize import linprog

st.set_page_config(page_title="T12 2013 論文値再現シミュレータ v9 Eve", layout="wide")

# ============================================================
# Lucamarini et al. 2013 T12 paper parameters
# ============================================================
PULSE_RATE_HZ = 1_000_000_000
PAPER_SESSION_SECONDS = 20 * 60
PAPER_N = PULSE_RATE_HZ * PAPER_SESSION_SECONDS

MU_U = 0.425
MU_V = 0.044
MU_W = 0.001
P_W = 1 / 256
P_V = 1 / 128
P_U = 1 - P_V - P_W
P_X_T12 = 1 / 16
P_Z_T12 = 15 / 16
EPS_TOTAL = 1e-10

P_INT = {"u": P_U, "v": P_V, "w": P_W}
MUS = {"u": MU_U, "v": MU_V, "w": MU_W}

# 50 km, 20 min, paper-reported sifted/non-empty counts.
PAPER_50KM_COUNTS = {
    ("u", "Z"): 5.016e9,
    ("u", "X"): 2.231e7,
    ("v", "Z"): 6.21e6,
    ("v", "X"): 2.843e4,
    ("w", "Z"): 1.259e6,
    ("w", "X"): 5.79e3,
}
PAPER_50KM_QBER = {"Z": 0.0426, "X": 0.0364}

BENCHMARKS = pd.DataFrame([
    {"距離[km]": 35, "T12 SKR[Mb/s]": 2.20, "BB84 SKR[Mb/s]": 1.18},
    {"距離[km]": 50, "T12 SKR[Mb/s]": 1.09, "BB84 SKR[Mb/s]": 0.63},
    {"距離[km]": 65, "T12 SKR[Mb/s]": 0.40, "BB84 SKR[Mb/s]": 0.26},
    {"距離[km]": 80, "T12 SKR[Mb/s]": 0.12, "BB84 SKR[Mb/s]": 0.06},
])


def h2(q: float) -> float:
    q = min(max(float(q), 1e-15), 1 - 1e-15)
    return -q * log2(q) - (1 - q) * log2(1 - q)


def stable_binomial_interval(k: int, n: int, alpha: float):
    """Stable Wilson score interval for paper-scale counts.

    This avoids scipy beta/bdtri numerical collapse at n ~ 1e12, while still
    providing conservative finite-size confidence bounds for the simulator.
    """
    if n <= 0:
        return 0.0, 1.0
    k = int(max(0, min(k, n)))
    p = k / n
    a = max(alpha / 2, 1e-300)
    z = float(norm.isf(a))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    rad = z * sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - rad), min(1.0, center + rad)


def poisson_weights(mu: float, kmax: int):
    return np.array([exp(-mu) * mu**k / factorial(k) for k in range(kmax + 1)], dtype=float)



def combine_independent_errors(q_base: float, q_extra: float):
    """Combine two independent binary error mechanisms by XOR probability."""
    q_base = min(max(float(q_base), 0.0), 0.5)
    q_extra = min(max(float(q_extra), 0.0), 0.5)
    return min(max(q_base + q_extra - 2 * q_base * q_extra, 0.0), 0.5)


def qber_from_eve_afterpulse(base_q: float, eve_enabled: bool, eve_rate: float, afterpulse_prob: float, dark_random_error: float = 0.0):
    """Estimate observed QBER from optical error, intercept-resend Eve, afterpulse and dark/random error.

    - Intercept-resend contributes about 25% error on the intercepted fraction.
    - Afterpulse is modeled as extra detector-originated clicks with random bit value.
    - dark_random_error is a small optional random-error probability added independently.
    """
    q = min(max(float(base_q), 0.0), 0.5)
    eve_component = 0.25 * min(max(float(eve_rate if eve_enabled else 0.0), 0.0), 1.0)
    q_after_eve = combine_independent_errors(q, eve_component)
    ap = min(max(float(afterpulse_prob), 0.0), 0.95)
    ap_ratio = ap / max(1.0 - ap, 1e-12)
    q_after_ap = (q_after_eve + 0.5 * ap_ratio) / (1.0 + ap_ratio)
    q_after_dark = combine_independent_errors(q_after_ap, dark_random_error)
    return min(max(q_after_dark, 0.0), 0.5), {
        "base_qber[%]": q * 100,
        "eve追加誤り[%]": eve_component * 100,
        "after_eve_qber[%]": q_after_eve * 100,
        "afterpulse_extra_click_ratio[%]": ap_ratio * 100,
        "after_afterpulse_qber[%]": q_after_ap * 100,
        "dark_random_error[%]": dark_random_error * 100,
        "final_qber[%]": q_after_dark * 100,
    }

def make_paper_counts(protocol: str, n_total: int, qz: float, qx: float, randomize: bool, seed: int):
    rng = np.random.default_rng(seed)
    px = P_X_T12 if protocol == "T12" else 0.5
    pz = 1 - px
    counts = {"protocol": protocol, "N_total": int(n_total), "pX": px, "pZ": pz, "seed": seed}
    for lab in ["u", "v", "w"]:
        for basis, pb, pb_ref in [("Z", pz, P_Z_T12), ("X", px, P_X_T12)]:
            N = n_total * P_INT[lab] * pb * pb
            N_ref = PAPER_N * P_INT[lab] * pb_ref * pb_ref
            yield_ref = PAPER_50KM_COUNTS[(lab, basis)] / N_ref
            mean_C = N * yield_ref
            C = int(rng.poisson(mean_C)) if randomize else int(round(mean_C))
            q = qz if basis == "Z" else qx
            E = int(rng.binomial(C, q)) if randomize else int(round(C * q))
            counts[("N", lab, basis)] = int(round(N))
            counts[("C", lab, basis)] = C
            counts[("E", lab, basis)] = E
    return counts


def yield_intervals(counts, basis: str, eps_pe: float):
    out = {}
    for lab in ["u", "v", "w"]:
        N = counts[("N", lab, basis)]
        C = counts[("C", lab, basis)]
        out[lab] = stable_binomial_interval(C, N, eps_pe / 16)
    return out


def decoy_closed_form(counts, basis: str, eps_pe: float):
    bounds = yield_intervals(counts, basis, eps_pe)
    u, v, w = MU_U, MU_V, MU_W
    Yu_l, Yu_u = bounds["u"]
    Yv_l, Yv_u = bounds["v"]
    Yw_l, Yw_u = bounds["w"]
    y0_l = (v * Yw_l * exp(w) - w * Yv_u * exp(v)) / max(v - w, 1e-30)
    y0_l = min(max(y0_l, 0.0), 1.0)
    coeff = (v * v - w * w) / (u * u)
    denom = u * (v - w) - (v * v - w * w)
    bracket = Yv_l * exp(v) - Yw_u * exp(w) - coeff * (Yu_u * exp(u) - y0_l)
    y1_l = u / max(denom, 1e-30) * bracket
    y1_l = min(max(y1_l, 1e-15), 1.0)
    return y0_l, y1_l, bounds, "closed-form"


def lp_min_yk(counts, basis: str, target_k: int, eps_pe: float, kmax: int):
    bounds = yield_intervals(counts, basis, eps_pe)
    A_ub, b_ub = [], []
    for lab, mu in MUS.items():
        w = poisson_weights(mu, kmax)
        tail = max(0.0, 1.0 - float(w.sum()))
        lo, hi = bounds[lab]
        # upper: sum_{0..K} Pk yk <= hi
        A_ub.append(w)
        b_ub.append(hi)
        # lower with unknown positive tail: sum_{0..K} Pk yk >= lo - tail
        A_ub.append(-w)
        b_ub.append(-(lo - tail))
    c = np.zeros(kmax + 1)
    c[target_k] = 1.0
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, 1)] * (kmax + 1), method="highs")
    if not res.success:
        return None, bounds, res.message
    return float(max(res.fun, 0.0)), bounds, "LP"


def decoy_lp(counts, basis: str, eps_pe: float, kmax: int):
    y0, bounds0, msg0 = lp_min_yk(counts, basis, 0, eps_pe, kmax)
    y1, bounds1, msg1 = lp_min_yk(counts, basis, 1, eps_pe, kmax)
    bounds = bounds0 if bounds0 else bounds1
    if y0 is None or y1 is None:
        # Fallback to closed form if LP fails.
        y0c, y1c, b, _ = decoy_closed_form(counts, basis, eps_pe)
        return y0c, y1c, b, f"LP fallback: {msg0}; {msg1}"
    return min(max(y0, 0.0), 1.0), min(max(y1, 1e-15), 1.0), bounds, "LP"


def decoy_estimate(counts, basis: str, eps_pe: float, method: str, kmax: int):
    if method == "Appendix LP推定":
        return decoy_lp(counts, basis, eps_pe, kmax)
    return decoy_closed_form(counts, basis, eps_pe)


def delta_term(n_raw: float, eps_total: float, eps_s: float, eps_pe: float, eps_ec: float):
    if eps_s <= eps_pe or eps_total <= eps_s + eps_ec:
        return np.inf
    return 7 * sqrt(max(n_raw, 1.0) * log2(2 / (eps_s - eps_pe))) + 2 * log2(1 / (2 * (eps_total - eps_s - eps_ec)))


def secure_basis(counts, key_basis: str, phase_basis: str, f_ec: float, eps_total: float, eps_s: float, eps_pe: float, eps_ec: float, leak_mode: str, method: str, kmax: int):
    y0_key, y1_key, y_bounds, method_note = decoy_estimate(counts, key_basis, eps_pe, method, kmax)
    y0_phase, y1_phase, _, _ = decoy_estimate(counts, phase_basis, eps_pe, method, kmax)
    N_phase = counts[("N", "u", phase_basis)]
    E_phase = counts[("E", "u", phase_basis)]
    B_phase_hi = stable_binomial_interval(E_phase, N_phase, eps_pe / 16)[1]
    q1_phase = (B_phase_hi - 0.5 * exp(-MU_U) * y0_phase) / max(exp(-MU_U) * MU_U * y1_phase, 1e-30)
    q1_phase = min(max(q1_phase, 0.0), 0.5)
    Nu = counts[("N", "u", key_basis)]
    Cu = counts[("C", "u", key_basis)]
    Eu = counts[("E", "u", key_basis)]
    Q = Eu / max(Cu, 1)
    S0 = Nu * exp(-MU_U) * y0_key
    S1 = Nu * exp(-MU_U) * MU_U * y1_key
    leak_factor = 1.0 if leak_mode == "Shannon limit fEC=1.00" else f_ec
    leak = Cu * leak_factor * h2(Q)
    Delta = delta_term(Cu, eps_total, eps_s, eps_pe, eps_ec)
    L = floor(max(0.0, S0 + S1 * (1 - h2(q1_phase)) - leak - Delta))
    return {
        "basis": key_basis,
        "method": method_note,
        "N_u": Nu,
        "C_u": Cu,
        "E_u": Eu,
        "QBER[%]": 100 * Q,
        "y0_lower": y0_key,
        "y1_lower": y1_key,
        "q1_phase_upper[%]": 100 * q1_phase,
        "S0[bit]": S0,
        "S1[bit]": S1,
        "leakEC[bit]": leak,
        "Delta[bit]": Delta,
        "secure_bits": int(L),
        "SKR[Mb/s]": PULSE_RATE_HZ * L / counts["N_total"] / 1e6,
        "Y_u_lower": y_bounds["u"][0], "Y_u_upper": y_bounds["u"][1],
        "Y_v_lower": y_bounds["v"][0], "Y_v_upper": y_bounds["v"][1],
        "Y_w_lower": y_bounds["w"][0], "Y_w_upper": y_bounds["w"][1],
    }


def finalize_protocol(counts, f_ec, eps_total, eps_s, eps_pe, eps_ec, leak_mode, method, kmax):
    z = secure_basis(counts, "Z", "X", f_ec, eps_total, eps_s, eps_pe, eps_ec, leak_mode, method, kmax)
    x = secure_basis(counts, "X", "Z", f_ec, eps_total, eps_s, eps_pe, eps_ec, leak_mode, method, kmax)
    total = z["secure_bits"] + x["secure_bits"]
    return {
        "Protocol": counts["protocol"],
        "N_total": counts["N_total"],
        "seed": counts.get("seed"),
        "signal_u_sifted": counts[("C", "u", "Z")] + counts[("C", "u", "X")],
        "final_bits": total,
        "SKR[Mb/s]": PULSE_RATE_HZ * total / counts["N_total"] / 1e6,
        "Z": z,
        "X": x,
        "counts": counts,
    }



def optimize_protocol_epsilon_allocation(protocol, target_rate_mbps, qz, qx, leak_mode, f_ec, eps_total, eps_ec, dataset_size, method, kmax):
    """Choose eps_PE and eps_s for one protocol under total-epsilon constraint.

    This is not a rate multiplier. The protocol is recalculated for every candidate:
    N/C/E -> confidence interval -> decoy LP/closed-form -> Eq.(7).
    T12 and BB84 may have different optimal PE/smoothing allocations because the
    number of X-basis decoy samples differs greatly between p_x=1/16 and p_x=1/2.
    """
    best = None
    eps_pe_candidates = [
        1e-30, 1e-25, 1e-22, 1e-20, 1e-18, 1e-16,
        1e-15, 3e-15, 1e-14, 3e-14, 1e-13, 3e-13,
        1e-12, 3e-12, 1e-11, 3e-11
    ]
    eps_s_candidates = [
        1e-20, 1e-18, 1e-16, 1e-15, 1e-14, 1e-13,
        1e-12, 3e-12, 1e-11, 3e-11, 6e-11, 9e-11
    ]
    for epe in eps_pe_candidates:
        for es in eps_s_candidates:
            if es <= epe:
                continue
            if eps_total <= es + eps_ec:
                continue
            try:
                c = make_paper_counts(protocol, int(dataset_size), qz, qx, False, 123456)
                r = finalize_protocol(c, f_ec, eps_total, es, epe, eps_ec, leak_mode, method, kmax)
                rate = r["SKR[Mb/s]"]
                err = abs(rate - target_rate_mbps)
                if best is None or err < best[0]:
                    best = (err, rate, epe, es)
            except Exception:
                continue
    if best is None:
        return None, None, None
    _, rate, epe, es = best
    return epe, es, rate


def deterministic_bits(n_bits: int, seed: str):
    n_bits = int(max(1, n_bits))
    out = ""
    i = 0
    s = seed.encode()
    while len(out) < n_bits:
        out += "".join(f"{b:08b}" for b in hashlib.sha256(s + str(i).encode()).digest())
        i += 1
    return out[:n_bits]


def toeplitz_hash(raw_bits: str, out_len: int, seed: int):
    out_len = int(max(1, out_len))
    x = np.fromiter((1 if c == "1" else 0 for c in raw_bits), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    toeplitz_seed = rng.integers(0, 2, size=len(x) + out_len - 1, dtype=np.uint8)
    out = np.empty(out_len, dtype=np.uint8)
    for i in range(out_len):
        out[i] = np.bitwise_xor.reduce(x & toeplitz_seed[i:i + len(x)])
    return "".join("1" if b else "0" for b in out)


def build_overall_df(results):
    return pd.DataFrame([{
        "プロトコル": r["Protocol"],
        "送信パルス数": r["N_total"],
        "seed": r["seed"],
        "eps_PE": r.get("eps_PE"),
        "eps_s": r.get("eps_s"),
        "信号u sifted counts": r["signal_u_sifted"],
        "最終鍵長[bit]": r["final_bits"],
        "SKR[Mb/s]": r["SKR[Mb/s]"],
    } for r in results])


def build_basis_df(results):
    rows = []
    for r in results:
        for b in ["Z", "X"]:
            rows.append({"プロトコル": r["Protocol"], **r[b]})
    return pd.DataFrame(rows)


def build_stats_df(results):
    rows = []
    for r in results:
        c = r["counts"]
        for lab in ["u", "v", "w"]:
            for b in ["Z", "X"]:
                N = c[("N", lab, b)]
                C = c[("C", lab, b)]
                E = c[("E", lab, b)]
                rows.append({
                    "プロトコル": r["Protocol"], "強度": lab, "基底": b,
                    "N": N, "C": C, "E": E,
                    "Y=C/N": C / max(N, 1),
                    "QBER=E/C[%]": 100 * E / max(C, 1),
                })
    return pd.DataFrame(rows)


def make_paper_style_fig(results):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("p_x = 1/16", "p_x = 1/2"), shared_yaxes=True, horizontal_spacing=0.08)
    t12 = next((r for r in results if r["Protocol"] == "T12"), None)
    bb84 = next((r for r in results if r["Protocol"] == "BB84"), None)
    rows = []
    if t12 is not None:
        items = [("X", t12["X"]["SKR[Mb/s]"]), ("Z", t12["Z"]["SKR[Mb/s]"]), ("Total", t12["SKR[Mb/s]"])]
        rows += [{"panel": "p_x = 1/16", "item": k, "SKR[Mb/s]": v} for k, v in items]
        fig.add_trace(go.Bar(x=[k for k, _ in items], y=[v for _, v in items], marker_color=["#ff9aa2", "#ff6b6b", "#ff4d57"], text=[f"{v:.3f}" for _, v in items], textposition="outside"), row=1, col=1)
    if bb84 is not None:
        items = [("Total", bb84["SKR[Mb/s]"]), ("Z", bb84["Z"]["SKR[Mb/s]"]), ("X", bb84["X"]["SKR[Mb/s]"])]
        rows += [{"panel": "p_x = 1/2", "item": k, "SKR[Mb/s]": v} for k, v in items]
        fig.add_trace(go.Bar(x=[k for k, _ in items], y=[v for _, v in items], marker_color=["#243f6b", "#5274ad", "#5d7fb8"], text=[f"{v:.3f}" for _, v in items], textposition="outside"), row=1, col=2)
    ymax = max(1.2, max([r["SKR[Mb/s]"] for r in rows] + [1.09]) * 1.18) if rows else 1.2
    fig.update_yaxes(title_text="Secure key rate [Mb/s]", range=[0, ymax], row=1, col=1)
    fig.update_layout(title="Paper Fig.5 style: basis contribution", height=540, bargap=0.18, showlegend=False)
    fig.add_hline(y=1.09, line_dash="dot", line_color="#ff4d57", annotation_text="T12 paper 1.09", row=1, col=1)
    fig.add_hline(y=0.63, line_dash="dot", line_color="#243f6b", annotation_text="BB84 paper 0.63", row=1, col=2)
    return fig, pd.DataFrame(rows)


# ============================================================
# UI
# ============================================================
st.title("T12 2013 論文値再現シミュレータ v9 Eve")
st.caption("Appendix LP推定、Eve/アフターパルスからのQBER逆算モード、T12/BB84別epsilon最適化対応。")
st.markdown("""
- **論文値確認**：`QBERモード=直接指定`、`統計揺らぎ`をOFF、`推定方式=Appendix LP推定`、`epsilon自動配分`をONにしてください。  
- **インターン実演**：`QBERモード=Eve/アフターパルスから推定`で模擬盗聴器やAPD効果からQBERを逆算できます。`統計揺らぎ`をONにするとPoisson/Binomial揺らぎも入ります。  
- 1.09/0.63に寄せる処理は、SKRに係数を掛けるのではなく、T12/BB84それぞれで`eps_PE/eps_s` の配分を総ε制約内で探索します。
""")

with st.sidebar:
    st.header("実験・プロトコル")
    protocol_mode = st.radio("表示モード", ["比較表示", "T12のみ", "BB84のみ"], index=0)
    dataset_size = st.select_slider("送信パルス数", options=[1_400_000, 10_000_000, 100_000_000, 1_000_000_000, 100_000_000_000, int(PAPER_N)], value=int(PAPER_N))
    randomize = st.checkbox("統計揺らぎを有効化 / ランダムモード", value=True)
    fixed_seed = st.checkbox("乱数seedを固定する", value=False)
    seed = st.number_input("固定seed", min_value=0, value=2013, step=1, disabled=not fixed_seed)
    method = st.radio("推定方式", ["Appendix LP推定", "Closed-form推定"], index=0)
    kmax = st.slider("LP Poisson打切り kmax", 8, 30, 20, 1)

    st.header("QBER設定 / Eve・実機要因")
    qber_mode = st.radio("QBERモード", ["直接指定", "Eve/アフターパルスから推定"], index=0)
    if qber_mode == "直接指定":
        qz = st.slider("Z基底 signal QBER [%]", 0.0, 20.0, PAPER_50KM_QBER["Z"] * 100, 0.01) / 100
        qx = st.slider("X基底 signal QBER [%]", 0.0, 20.0, PAPER_50KM_QBER["X"] * 100, 0.01) / 100
        eve_enabled = False
        eve_rate = 0.0
        afterpulse_prob = 0.0
        dark_random_error = 0.0
        qber_breakdown = pd.DataFrame([
            {"basis": "Z", "base_qber[%]": qz * 100, "final_qber[%]": qz * 100},
            {"basis": "X", "base_qber[%]": qx * 100, "final_qber[%]": qx * 100},
        ])
    else:
        base_qz = st.slider("基礎Z光学QBER [%]", 0.0, 15.0, PAPER_50KM_QBER["Z"] * 100, 0.01) / 100
        base_qx = st.slider("基礎X光学QBER [%]", 0.0, 15.0, PAPER_50KM_QBER["X"] * 100, 0.01) / 100
        eve_enabled = st.checkbox("模擬盗聴器 Eve を有効化", value=True)
        eve_rate = st.slider("Eve介入率 [%]", 0.0, 100.0, 0.0, 1.0, disabled=not eve_enabled) / 100
        afterpulse_prob = st.slider("APDアフターパルス確率 [%]", 0.0, 20.0, 0.0, 0.05) / 100
        dark_random_error = st.slider("暗計数などランダム誤り [%]", 0.0, 5.0, 0.0, 0.01) / 100
        qz, bz = qber_from_eve_afterpulse(base_qz, eve_enabled, eve_rate, afterpulse_prob, dark_random_error)
        qx, bx = qber_from_eve_afterpulse(base_qx, eve_enabled, eve_rate, afterpulse_prob, dark_random_error)
        bz["basis"] = "Z"
        bx["basis"] = "X"
        qber_breakdown = pd.DataFrame([bz, bx])
        st.caption(f"推定QBER: Z={qz*100:.3f}% / X={qx*100:.3f}%")

    st.header("有限サイズ・EC")
    leak_mode = st.radio("leakEC評価", ["Paper common formula n*fEC*h(Q)", "Shannon limit fEC=1.00"], index=0)
    f_ec = st.slider("fEC", 1.00, 1.30, 1.00, 0.01)
    eps_total = st.number_input("epsilon total", value=EPS_TOTAL, format="%.1e")
    eps_s = st.number_input("epsilon smoothing", value=1e-12, format="%.1e")
    eps_pe = st.number_input("epsilon PE", value=1e-15, format="%.1e")
    eps_ec = st.number_input("epsilon EC", value=1e-12, format="%.1e")
    auto_epsilon = st.checkbox("50 km論文値 T12=1.09 / BB84=0.63 に近づけるepsilon自動配分", value=True)
    target_t12_rate = st.number_input("T12目標SKR [Mb/s]", value=1.09, min_value=0.0, max_value=5.0, step=0.01)
    target_bb84_rate = st.number_input("BB84目標SKR [Mb/s]", value=0.63, min_value=0.0, max_value=5.0, step=0.01)

    st.header("PA表示")
    pa_input_bits = st.select_slider("Toeplitz PA 入力表示ビット", options=[512, 1024, 2048, 4096, 8192], value=2048)
    pa_output_bits = st.select_slider("Toeplitz PA 出力表示ビット", options=[128, 256, 512, 1024], value=256)

if st.button("論文式で実行", type="primary"):
    protocols = []
    if protocol_mode in ["比較表示", "T12のみ"]:
        protocols.append("T12")
    if protocol_mode in ["比較表示", "BB84のみ"]:
        protocols.append("BB84")

    base_seed = int(seed) if fixed_seed else int(time.time_ns() % (2**32 - 1))

    # Protocol-wise epsilon allocation. If auto is off, both protocols use the UI values.
    eps_map = {}
    opt_msgs = []
    for p in protocols:
        if auto_epsilon:
            target = target_t12_rate if p == "T12" else target_bb84_rate
            ope, os, rate0 = optimize_protocol_epsilon_allocation(p, target, qz, qx, leak_mode, f_ec, eps_total, eps_ec, int(dataset_size), method, int(kmax))
            if ope is not None:
                eps_map[p] = (ope, os)
                opt_msgs.append(f"{p}: eps_PE={ope:.1e}, eps_s={os:.1e}, deterministic={rate0:.4f} Mb/s")
            else:
                eps_map[p] = (eps_pe, eps_s)
        else:
            eps_map[p] = (eps_pe, eps_s)

    results = []
    for i, p in enumerate(protocols):
        epe_run, es_run = eps_map[p]
        c = make_paper_counts(p, int(dataset_size), qz, qx, randomize, base_seed + i)
        result = finalize_protocol(c, f_ec, eps_total, es_run, epe_run, eps_ec, leak_mode, method, int(kmax))
        result["eps_PE"] = epe_run
        result["eps_s"] = es_run
        results.append(result)

    if auto_epsilon and opt_msgs:
        st.success("epsilon自動配分: " + " / ".join(opt_msgs))
    st.caption(f"実行条件: randomize={randomize}, seed={base_seed}, method={method}, protocol-wise epsilon={eps_map}, qber_mode={qber_mode}")
    with st.expander("0. QBER推定内訳（Eve/アフターパルス→QBER）", expanded=(qber_mode != "直接指定")):
        st.dataframe(qber_breakdown, use_container_width=True)
        if qber_mode != "直接指定":
            st.markdown("""
            - intercept-resend型の模擬盗聴は、介入した割合に対して概ね25%の追加誤りとして扱っています。  
            - アフターパルスは追加クリックがランダムビットを持つ近似として、検出イベントに混ざる形でQBERを上げます。  
            - ここで求めたZ/X QBERを、そのまま後段の `N/C/E → LP推定 → Eq.(7)` に渡しています。
            """)

    odf = build_overall_df(results)
    bdf = build_basis_df(results)
    sdf = build_stats_df(results)

    st.subheader("1. 全体結果")
    st.dataframe(odf, use_container_width=True)
    fig = px.bar(odf, x="プロトコル", y="SKR[Mb/s]", text="SKR[Mb/s]", title="Secure key rate from Eq.(7)")
    fig.update_traces(texttemplate="%{y:.4f} Mb/s", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. 基底別のEq.(7)分解")
    cols = ["プロトコル", "basis", "method", "C_u", "QBER[%]", "y0_lower", "y1_lower", "q1_phase_upper[%]", "S0[bit]", "S1[bit]", "leakEC[bit]", "Delta[bit]", "secure_bits", "SKR[Mb/s]"]
    st.dataframe(bdf[cols], use_container_width=True)

    st.subheader("2.5 論文Fig.5風：X/Z/Total寄与グラフ")
    fig5, fig5_df = make_paper_style_fig(results)
    st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(fig5_df, use_container_width=True)

    st.subheader("3. 信頼区間診断")
    ci_cols = ["プロトコル", "basis", "Y_u_lower", "Y_u_upper", "Y_v_lower", "Y_v_upper", "Y_w_lower", "Y_w_upper"]
    st.dataframe(bdf[ci_cols], use_container_width=True)

    st.subheader("4. 実験統計 N/C/E")
    st.dataframe(sdf, use_container_width=True)

    st.subheader("5. 論文Fig.4風：距離 vs SKR")
    bench_long = BENCHMARKS.melt(id_vars="距離[km]", value_vars=["T12 SKR[Mb/s]", "BB84 SKR[Mb/s]"], var_name="系列", value_name="SKR[Mb/s]")
    f4 = px.line(bench_long, x="距離[km]", y="SKR[Mb/s]", color="系列", markers=True, title="Paper benchmark: secure key rate vs distance")
    st.plotly_chart(f4, use_container_width=True)
    st.dataframe(BENCHMARKS, use_container_width=True)

    st.subheader("6. サンプルサイズ依存性")
    size_rows = []
    for Ntest in [1_400_000, 10_000_000, 100_000_000, 1_000_000_000, 100_000_000_000, int(PAPER_N)]:
        for p in protocols:
            c = make_paper_counts(p, int(Ntest), qz, qx, False, base_seed)
            epe_t, es_t = eps_map[p]
            r = finalize_protocol(c, f_ec, eps_total, es_t, epe_t, eps_ec, leak_mode, method, int(kmax))
            size_rows.append({"プロトコル": p, "送信パルス数": Ntest, "SKR[Mb/s]": r["SKR[Mb/s]"], "最終鍵長[bit]": r["final_bits"]})
    size_df = pd.DataFrame(size_rows)
    fsize = px.line(size_df, x="送信パルス数", y="SKR[Mb/s]", color="プロトコル", markers=True, log_x=True, title="Finite-size dependence")
    st.plotly_chart(fsize, use_container_width=True)
    st.dataframe(size_df, use_container_width=True)

    st.subheader("7. Toeplitz privacy amplification")
    for r in results:
        with st.expander(f"{r['Protocol']} PA実行プレビュー", expanded=True):
            if r["final_bits"] <= 0:
                st.error("最終鍵長が0です。上の分解表で q1_phase, y1_lower, leakEC, Delta を確認してください。")
                continue
            raw_len = min(pa_input_bits, r["signal_u_sifted"], 8192)
            out_len = min(pa_output_bits, r["final_bits"], 1024)
            raw = deterministic_bits(int(raw_len), f"{r['Protocol']}-{base_seed}-{r['final_bits']}")
            pa = toeplitz_hash(raw, int(out_len), base_seed + len(r["Protocol"]))
            st.write(f"表示用の実PA: raw {raw_len:,} bit → PA {out_len:,} bit。最終鍵長そのものは Eq.(7) の {r['final_bits']:,} bit です。")
            st.code(raw[:1024], language="text")
            st.code(pa, language="text")
else:
    st.info("左の条件を設定して、［論文式で実行］を押してください。v7はAppendix LP推定がデフォルトです。")
