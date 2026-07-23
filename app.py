import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import hashlib
from math import exp, log2, sqrt, floor, log10

# SciPy is required for robust high-quantile normal values.
# Exact beta-PPF Clopper-Pearson is intentionally NOT used for 1e12-scale n,
# because scipy.stats.beta.ppf can numerically collapse to 1.0 for huge alpha/beta.
from scipy.stats import norm
from scipy.special import bdtri

st.set_page_config(page_title="T12 2013 Paper Exact Reproduction v2", layout="wide")

# ============================================================
# Lucamarini et al. 2013, Optics Express 21, 24550-24565
# Efficient decoy-state quantum key distribution with quantified security
# ============================================================
PULSE_RATE_HZ = 1_000_000_000
PAPER_SESSION_SECONDS = 20 * 60
PAPER_N = PULSE_RATE_HZ * PAPER_SESSION_SECONDS

# T12 parameters explicitly stated in the paper
MU_U = 0.425
MU_V = 0.044
MU_W = 0.001
P_W = 1 / 256
P_V = 1 / 128
P_U = 1 - P_V - P_W
P_X_T12 = 1 / 16
P_Z_T12 = 15 / 16
EPS_TOTAL = 1e-10

MUS = {"u": MU_U, "v": MU_V, "w": MU_W}
P_INT = {"u": P_U, "v": P_V, "w": P_W}

# Paper 50 km, 20 min measured sifted counts, Fig.3 text/table area.
# These are NOT fitted coefficients. They are the directly reported experimental counts.
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


def fmt_si(x, suffix=""):
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    for s, u in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")]:
        if x >= s:
            return f"{sign}{x/s:.3f}{u}{suffix}"
    return f"{sign}{x:.3f}{suffix}"


def robust_binomial_interval(k: int, n: int, alpha: float):
    """Clopper-Pearson interval with numerical-stability guard.

    The paper uses Clopper-Pearson confidence intervals. For paper-scale n,
    beta.ppf can return 1.0 due to numerical collapse, so this implementation
    uses scipy.special.bdtri, which is equivalent to CP inversion of the binomial
    CDF. If bdtri fails, it falls back to Wilson only as a diagnostic guard.
    """
    if n <= 0:
        return 0.0, 1.0
    k = int(max(0, min(k, n)))
    a = max(alpha / 2, 1e-300)
    try:
        lo = 0.0 if k == 0 else float(bdtri(k - 1, n, 1 - a))
        hi = 1.0 if k == n else float(bdtri(k, n, a))
        if np.isfinite(lo) and np.isfinite(hi) and 0 <= lo <= hi <= 1 and not (hi == 1.0 and k < n):
            return lo, hi
    except Exception:
        pass
    p = k / n
    z = float(norm.isf(a))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    rad = z * sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - rad), min(1.0, center + rad)


def make_paper_counts(protocol: str, n_total: int, qz: float, qx: float, randomize: bool, seed: int):
    """Generate N/C/E tables from paper-measured 50 km yields.

    For T12 at N=1.2e12, this reproduces the paper's reported measured counts.
    For other N, it uses the same empirical yields and changes only the session length.
    For BB84, it uses the same optical yields but pX=pZ=1/2.
    This is not coefficient-fitting to match SKR; it is yield-based reconstruction.
    """
    rng = np.random.default_rng(seed)
    px = P_X_T12 if protocol == "T12" else 0.5
    pz = 1 - px
    counts = {"protocol": protocol, "N_total": int(n_total), "pX": px, "pZ": pz}
    for lab in ["u", "v", "w"]:
        for basis, pb, pb_ref in [("Z", pz, P_Z_T12), ("X", px, P_X_T12)]:
            N = n_total * P_INT[lab] * pb * pb
            N_ref = PAPER_N * P_INT[lab] * pb_ref * pb_ref
            yield_ref = PAPER_50KM_COUNTS[(lab, basis)] / N_ref
            mean_C = N * yield_ref
            C = rng.poisson(mean_C) if randomize else int(round(mean_C))
            q = qz if basis == "Z" else qx
            E = rng.binomial(C, q) if randomize else int(round(C * q))
            counts[("N", lab, basis)] = int(round(N))
            counts[("C", lab, basis)] = int(C)
            counts[("E", lab, basis)] = int(E)
    return counts


def decoy_bounds_basis(counts, basis: str, eps_pe: float):
    """3-intensity decoy lower bounds for y0 and y1, using bounded observed yields."""
    bounds = {}
    for lab in ["u", "v", "w"]:
        N = counts[("N", lab, basis)]
        C = counts[("C", lab, basis)]
        bounds[lab] = robust_binomial_interval(C, N, eps_pe / 16)

    u, v, w = MU_U, MU_V, MU_W
    Yu_l, Yu_u = bounds["u"]
    Yv_l, Yv_u = bounds["v"]
    Yw_l, Yw_u = bounds["w"]

    # Vacuum lower bound from v/w.
    y0_l = (v * Yw_l * exp(w) - w * Yv_u * exp(v)) / max(v - w, 1e-30)
    y0_l = min(max(y0_l, 0.0), 1.0)

    # Single-photon lower bound using u/v/w, finite statistics worst case.
    coeff = (v * v - w * w) / (u * u)
    denom = u * (v - w) - (v * v - w * w)
    bracket = Yv_l * exp(v) - Yw_u * exp(w) - coeff * (Yu_u * exp(u) - y0_l)
    y1_l = u / max(denom, 1e-30) * bracket
    y1_l = min(max(y1_l, 1e-15), 1.0)

    return y0_l, y1_l, bounds


def delta_term(n_raw: float, eps_total: float, eps_s: float, eps_pe: float, eps_ec: float):
    # Paper Eq.(3)-style finite-size term as written in the article text.
    if eps_s <= eps_pe or eps_total <= eps_s + eps_ec:
        return np.inf
    return 7 * sqrt(max(n_raw, 1.0) * log2(2 / (eps_s - eps_pe))) + 2 * log2(1 / (2 * (eps_total - eps_s - eps_ec)))


def secure_basis(counts, key_basis: str, phase_basis: str, f_ec: float, eps_total: float, eps_s: float, eps_pe: float, eps_ec: float, leak_mode: str):
    y0_key, y1_key, y_bounds = decoy_bounds_basis(counts, key_basis, eps_pe)
    y0_phase, y1_phase, phase_bounds = decoy_bounds_basis(counts, phase_basis, eps_pe)

    N_phase_u = counts[("N", "u", phase_basis)]
    E_phase_u = counts[("E", "u", phase_basis)]
    B_phase_hi = robust_binomial_interval(E_phase_u, N_phase_u, eps_pe / 16)[1]

    q1_phase = (B_phase_hi - 0.5 * exp(-MU_U) * y0_phase) / max(exp(-MU_U) * MU_U * y1_phase, 1e-30)
    q1_phase = min(max(q1_phase, 0.0), 0.5)

    Nu = counts[("N", "u", key_basis)]
    Cu = counts[("C", "u", key_basis)]
    Eu = counts[("E", "u", key_basis)]
    Q = Eu / max(Cu, 1)

    S0 = Nu * exp(-MU_U) * y0_key
    S1 = Nu * exp(-MU_U) * MU_U * y1_key

    if leak_mode == "Shannon limit fEC=1.00":
        leak = Cu * 1.00 * h2(Q)
    else:
        leak = Cu * f_ec * h2(Q)

    Delta = delta_term(Cu, eps_total, eps_s, eps_pe, eps_ec)
    L = floor(max(0.0, S0 + S1 * (1 - h2(q1_phase)) - leak - Delta))
    return {
        "basis": key_basis,
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
        "Y_u_lower": y_bounds["u"][0],
        "Y_u_upper": y_bounds["u"][1],
        "Y_v_lower": y_bounds["v"][0],
        "Y_v_upper": y_bounds["v"][1],
        "Y_w_lower": y_bounds["w"][0],
        "Y_w_upper": y_bounds["w"][1],
    }


def finalize_protocol(counts, f_ec, eps_total, eps_s, eps_pe, eps_ec, leak_mode):
    z = secure_basis(counts, "Z", "X", f_ec, eps_total, eps_s, eps_pe, eps_ec, leak_mode)
    x = secure_basis(counts, "X", "Z", f_ec, eps_total, eps_s, eps_pe, eps_ec, leak_mode)
    total = z["secure_bits"] + x["secure_bits"]
    return {
        "Protocol": counts["protocol"],
        "N_total": counts["N_total"],
        "signal_u_sifted": counts[("C", "u", "Z")] + counts[("C", "u", "X")],
        "final_bits": total,
        "SKR[Mb/s]": PULSE_RATE_HZ * total / counts["N_total"] / 1e6,
        "Z": z,
        "X": x,
        "counts": counts,
    }


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
    """Actual Toeplitz universal hashing for the displayed block."""
    out_len = int(max(1, out_len))
    x = np.fromiter((1 if c == "1" else 0 for c in raw_bits), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    toeplitz_seed = rng.integers(0, 2, size=len(x) + out_len - 1, dtype=np.uint8)
    out = np.empty(out_len, dtype=np.uint8)
    for i in range(out_len):
        out[i] = np.bitwise_xor.reduce(x & toeplitz_seed[i:i+len(x)])
    return "".join("1" if b else "0" for b in out)


def overall_df(results):
    return pd.DataFrame([{
        "プロトコル": r["Protocol"],
        "送信パルス数": r["N_total"],
        "信号u sifted counts": r["signal_u_sifted"],
        "最終鍵長[bit]": r["final_bits"],
        "SKR[Mb/s]": r["SKR[Mb/s]"],
    } for r in results])


def basis_df(results):
    rows = []
    for r in results:
        for b in ["Z", "X"]:
            rows.append({"プロトコル": r["Protocol"], **r[b]})
    return pd.DataFrame(rows)


def stats_df(results):
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


st.title("T12 2013 論文値再現シミュレータ v2")
st.caption("係数合わせなし。論文の50 km実測カウントから、有限サイズデコイ推定→Eq.(7)→Toeplitz PA表示まで通します。")

st.markdown("""
今回のv2では、前回の **論文値モードで0になる原因** を潰しています。  
原因は、`beta.ppf` によるClopper-Pearson実装が、`n ≈ 10^12` 規模で上限を `1.0` と返す数値破綻を起こし、
その結果 `q1_phase = 50%` まで悪化して、Eq.(7)の単一光子項が消えていたことです。  
この版では、巨大サンプルに対して数値安定なWilson型信頼区間を使い、論文と同じ流れで鍵長を出します。
""")

with st.sidebar:
    st.header("実験・プロトコル")
    protocol_mode = st.radio("表示モード", ["比較表示", "T12のみ", "BB84のみ"], index=0)
    dataset_size = st.select_slider(
        "送信パルス数",
        options=[1_400_000, 10_000_000, 100_000_000, 1_000_000_000, 100_000_000_000, int(PAPER_N)],
        value=int(PAPER_N),
    )
    randomize = st.checkbox("統計揺らぎをサンプリング", value=False)
    seed = st.number_input("乱数seed", min_value=0, value=2013, step=1)

    st.header("50 km論文値")
    qz = st.slider("Z基底 signal QBER [%]", 0.0, 15.0, PAPER_50KM_QBER["Z"] * 100, 0.01) / 100
    qx = st.slider("X基底 signal QBER [%]", 0.0, 15.0, PAPER_50KM_QBER["X"] * 100, 0.01) / 100

    st.header("有限サイズ・EC")
    leak_mode = st.radio("leakEC評価", ["Paper common formula n*fEC*h(Q)", "Shannon limit fEC=1.00"], index=0)
    f_ec = st.slider("fEC", 1.00, 1.30, 1.00, 0.01)
    eps_total = st.number_input("epsilon total", value=EPS_TOTAL, format="%.1e")
    eps_s = st.number_input("epsilon smoothing", value=1e-12, format="%.1e")
    eps_pe = st.number_input("epsilon PE", value=1e-15, format="%.1e")
    eps_ec = st.number_input("epsilon EC", value=1e-12, format="%.1e")

    st.header("PA表示")
    pa_input_bits = st.select_slider("Toeplitz PA 入力表示ビット", options=[512, 1024, 2048, 4096, 8192], value=2048)
    pa_output_bits = st.select_slider("Toeplitz PA 出力表示ビット", options=[128, 256, 512, 1024], value=256)

if st.button("論文式で実行", type="primary"):
    protocols = []
    if protocol_mode in ["比較表示", "T12のみ"]:
        protocols.append("T12")
    if protocol_mode in ["比較表示", "BB84のみ"]:
        protocols.append("BB84")

    results = []
    for i, p in enumerate(protocols):
        c = make_paper_counts(p, int(dataset_size), qz, qx, randomize, int(seed) + i)
        results.append(finalize_protocol(c, f_ec, eps_total, eps_s, eps_pe, eps_ec, leak_mode))

    odf = overall_df(results)
    bdf = basis_df(results)
    sdf = stats_df(results)

    st.subheader("1. 全体結果")
    st.dataframe(odf, use_container_width=True)
    fig = px.bar(odf, x="プロトコル", y="SKR[Mb/s]", text="SKR[Mb/s]", title="Secure key rate from Eq.(7)")
    fig.update_traces(texttemplate="%{y:.3f} Mb/s", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. 基底別のEq.(7)分解")
    show_cols = ["プロトコル", "basis", "C_u", "QBER[%]", "y0_lower", "y1_lower", "q1_phase_upper[%]", "S0[bit]", "S1[bit]", "leakEC[bit]", "Delta[bit]", "secure_bits", "SKR[Mb/s]"]
    st.dataframe(bdf[show_cols], use_container_width=True)
    fig2 = px.bar(bdf, x="プロトコル", y="secure_bits", color="basis", barmode="group", text="secure_bits", title="Z/X basis key contribution")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. 信頼区間診断")
    ci_cols = ["プロトコル", "basis", "Y_u_lower", "Y_u_upper", "Y_v_lower", "Y_v_upper", "Y_w_lower", "Y_w_upper"]
    st.dataframe(bdf[ci_cols], use_container_width=True)
    st.caption("ここで Y_u_upper が 1.0 になる場合は統計計算が壊れています。v2ではここが小さい値に戻るよう修正しています。")

    st.subheader("4. 実験統計 N/C/E")
    st.dataframe(sdf, use_container_width=True)

    st.subheader("5. 論文のFig.4風グラフ")
    bench_long = BENCHMARKS.melt(id_vars="距離[km]", value_vars=["T12 SKR[Mb/s]", "BB84 SKR[Mb/s]"], var_name="系列", value_name="SKR[Mb/s]")
    f4 = px.line(bench_long, x="距離[km]", y="SKR[Mb/s]", color="系列", markers=True, title="Paper benchmark: secure key rate vs distance")
    st.plotly_chart(f4, use_container_width=True)
    st.dataframe(BENCHMARKS, use_container_width=True)

    st.subheader("6. サンプルサイズ依存性")
    size_rows = []
    for Ntest in [1_400_000, 10_000_000, 100_000_000, 1_000_000_000, 100_000_000_000, int(PAPER_N)]:
        for p in protocols:
            c = make_paper_counts(p, int(Ntest), qz, qx, False, int(seed))
            r = finalize_protocol(c, f_ec, eps_total, eps_s, eps_pe, eps_ec, leak_mode)
            size_rows.append({"プロトコル": p, "送信パルス数": Ntest, "SKR[Mb/s]": r["SKR[Mb/s]"], "最終鍵長[bit]": r["final_bits"]})
    size_df = pd.DataFrame(size_rows)
    f5 = px.line(size_df, x="送信パルス数", y="SKR[Mb/s]", color="プロトコル", markers=True, log_x=True, title="Finite-size dependence")
    st.plotly_chart(f5, use_container_width=True)
    st.dataframe(size_df, use_container_width=True)

    st.subheader("7. Toeplitz privacy amplification")
    for r in results:
        with st.expander(f"{r['Protocol']} PA実行プレビュー", expanded=True):
            if r["final_bits"] <= 0:
                st.error("最終鍵長が0です。上の分解表で q1_phase, y1_lower, leakEC, Delta を確認してください。")
                continue
            raw_len = min(pa_input_bits, r["signal_u_sifted"], 8192)
            out_len = min(pa_output_bits, r["final_bits"], 1024)
            raw = deterministic_bits(int(raw_len), f"{r['Protocol']}-{seed}-{r['final_bits']}")
            pa = toeplitz_hash(raw, int(out_len), int(seed) + len(r["Protocol"]))
            st.write(f"表示用の実PA: raw {raw_len:,} bit → PA {out_len:,} bit。最終鍵長そのものは Eq.(7) の {r['final_bits']:,} bit です。")
            st.code(raw[:1024], language="text")
            st.code(pa, language="text")
else:
    st.info("左の条件を設定して、［論文式で実行］を押してください。デフォルトは 50 km・20分・T12論文値です。")
