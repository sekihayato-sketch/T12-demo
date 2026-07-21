import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
import hashlib

st.set_page_config(page_title="T12 2013 Paper Reproduction Simulator", layout="wide")

# ============================================================
# Lucamarini et al. 2013 T12 paper parameters
# Efficient decoy-state quantum key distribution with quantified security
# ============================================================
PULSE_RATE_HZ = 1_000_000_000

# Intensities: u, v, w
MU_U = 0.425
MU_V = 0.044
MU_W = 0.001

# Intensity probabilities: p_w = 1/256, p_v = 1/128, p_u = 1 - p_v - p_w
P_W = 1 / 256
P_V = 1 / 128
P_U = 1.0 - P_V - P_W

# Basis probabilities: p_X = 1/16, p_Z = 15/16
P_X_T12 = 1 / 16
P_Z_T12 = 1.0 - P_X_T12

# Security parameter used in paper
EPS_TOTAL = 1e-10

# ============================================================
# Default UI values for 2013 paper reproduction around 50 km
# ============================================================
DEFAULT_DATASET_SIZE = 100_660_000
DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_PROTOCOL_MODE = "比較表示"

# Experimental detector values from the 2013 paper
DEFAULT_SPD_EFF_PERCENT = 20.5
DEFAULT_DARK_COUNT_PERCENT = 0.0021  # 2.1e-5 per gate = 0.0021% / pulse
DEFAULT_AFTERPULSE_PERCENT = 0.0

# 50 km fibre at ~0.2 dB/km -> 10 dB. Receiver loss is an effective optical loss.
DEFAULT_CHANNEL_LOSS_DB = 10.0
DEFAULT_RX_LOSS_DB = 2.6

# The paper reports QBER around 4.26% Z and 3.64% X at 50 km.
DEFAULT_EOPT_PERCENT = 3.9

DEFAULT_QBER_THRESHOLD_PERCENT = 11.0
DEFAULT_FINITE_SIGMA = 1.0
DEFAULT_EC_BLOCK_SIZE = 1_000_000
DEFAULT_QBER_SAMPLE_BITS = 8192
DEFAULT_F_EC = 1.10
DEFAULT_EC_FAIL_PERCENT = 0.0

BENCHMARKS = pd.DataFrame([
    {"距離[km]": 35, "T12 SKR[Mb/s]": 2.20, "BB84 SKR[Mb/s]": 1.18},
    {"距離[km]": 50, "T12 SKR[Mb/s]": 1.09, "BB84 SKR[Mb/s]": 0.63},
    {"距離[km]": 65, "T12 SKR[Mb/s]": 0.40, "BB84 SKR[Mb/s]": 0.26},
    {"距離[km]": 80, "T12 SKR[Mb/s]": 0.12, "BB84 SKR[Mb/s]": 0.06},
])


def reset_defaults():
    st.session_state["dataset_size"] = DEFAULT_DATASET_SIZE
    st.session_state["chunk_size"] = DEFAULT_CHUNK_SIZE
    st.session_state["protocol_mode"] = DEFAULT_PROTOCOL_MODE
    st.session_state["spd_eff_percent"] = DEFAULT_SPD_EFF_PERCENT
    st.session_state["rx_loss_db"] = DEFAULT_RX_LOSS_DB
    st.session_state["channel_loss_db"] = DEFAULT_CHANNEL_LOSS_DB
    st.session_state["dark_count_percent"] = DEFAULT_DARK_COUNT_PERCENT
    st.session_state["eopt_percent"] = DEFAULT_EOPT_PERCENT
    st.session_state["afterpulse_percent"] = DEFAULT_AFTERPULSE_PERCENT
    st.session_state["qber_threshold_percent"] = DEFAULT_QBER_THRESHOLD_PERCENT
    st.session_state["finite_sigma"] = DEFAULT_FINITE_SIGMA
    st.session_state["ec_block_size"] = DEFAULT_EC_BLOCK_SIZE
    st.session_state["qber_sample_bits"] = DEFAULT_QBER_SAMPLE_BITS
    st.session_state["f_ec"] = DEFAULT_F_EC
    st.session_state["ec_fail_percent"] = DEFAULT_EC_FAIL_PERCENT
    st.session_state["eve_enabled"] = False
    st.session_state["eve_rate_percent"] = 0


st.title("T12 2013論文シミュレータ")
st.caption("Lucamarini et al. 2013 のT12有限サイズ・デコイ状態QKD論文に合わせた教育用シミュレータです。固定PA圧縮率0.292は使いません。アフターパルスは実装依存の時間相関なので、理論再現デフォルトでは0%にしています。")

st.markdown("""
このアプリは、**Efficient decoy-state quantum key distribution with quantified security** のT12プロトコルを再現するための教育用シミュレータです。
以前の `PA圧縮率0.292` による上限制御は使わず、論文の式に近い形で、
**基底別・強度別の検出統計からデコイ推定を行い、Z基底とX基底の安全鍵率を別々に計算して合算**します。

### デフォルトで再現したい代表値
- 50 km光ファイバ相当
- T12 secure key rate: 約 **1.09 Mb/s**
- BB84 secure key rate: 約 **0.63 Mb/s**

### 2013 T12論文パラメータ
```text
pX = 1/16, pZ = 15/16
u = 0.425, v = 0.044, w = 0.001
p_u = 253/256, p_v = 1/128, p_w = 1/256
SPD efficiency = 20.5%
dark count probability = 2.1e-5 / gate
APD afterpulse probability = 5.25%（実機特性。理論再現デフォルトでは時間相関を直接モデル化しないため0%）
epsilon = 1e-10
```
""")

with st.sidebar:
    st.header("シミュレーション条件")
    if st.button("2013論文デフォルトに戻す"):
        reset_defaults()
        st.rerun()

    dataset_size = st.select_slider(
        "送信パルス数",
        options=[1_048_576, 4_194_304, 16_777_216, 33_554_432, 67_108_864, 100_660_000],
        value=DEFAULT_DATASET_SIZE,
        key="dataset_size",
    )
    protocol_mode = st.radio(
        "表示モード",
        ["比較表示", "BB84参照のみ", "T12論文値のみ"],
        index=0,
        key="protocol_mode",
    )
    chunk_size = st.select_slider(
        "内部処理チャンクサイズ",
        options=[250_000, 500_000, 1_000_000, 2_000_000, 5_000_000],
        value=DEFAULT_CHUNK_SIZE,
        key="chunk_size",
    )

    st.markdown("---")
    st.subheader("物理・検出条件")
    spd_efficiency = st.slider("SPD効率 [%]", 1.0, 100.0, DEFAULT_SPD_EFF_PERCENT, step=0.1, key="spd_eff_percent") / 100.0
    rx_loss_db = st.slider("受信光学損失 [dB]", 0.0, 15.0, DEFAULT_RX_LOSS_DB, step=0.1, key="rx_loss_db")
    channel_loss_db = st.slider("チャネル損失 [dB]", 0.0, 30.0, DEFAULT_CHANNEL_LOSS_DB, step=0.1, key="channel_loss_db")
    total_detection_efficiency = spd_efficiency * 10 ** (-(rx_loss_db + channel_loss_db) / 10)
    st.caption(f"総合検出効率 η = {total_detection_efficiency * 100:.4f}%")

    dark_count_rate = st.slider("暗計数率 Y0 [%/pulse]", 0.0, 0.1, DEFAULT_DARK_COUNT_PERCENT, step=0.0001, key="dark_count_percent") / 100.0
    optical_error_rate = st.slider("光学・変調誤差 Eopt [%]", 0.0, 10.0, DEFAULT_EOPT_PERCENT, step=0.1, key="eopt_percent") / 100.0
    afterpulse_percent = st.slider("アフターパルス確率 [%]", 0.0, 20.0, DEFAULT_AFTERPULSE_PERCENT, step=0.1, key="afterpulse_percent") / 100.0

    st.markdown("---")
    st.subheader("Eve")
    eve_enabled = st.checkbox("遮断・再送信攻撃を有効化", value=False, key="eve_enabled")
    eve_rate = st.slider("Eve介入率 [%]", 0, 100, 0, step=5, disabled=not eve_enabled, key="eve_rate_percent") / 100.0

    st.markdown("---")
    st.subheader("後処理")
    qber_threshold = st.slider("鍵破棄しきい値 QBER [%]", 0.0, 20.0, DEFAULT_QBER_THRESHOLD_PERCENT, step=0.5, key="qber_threshold_percent")
    finite_sigma = st.slider("有限サイズ補正 sigma", 0.0, 10.0, DEFAULT_FINITE_SIGMA, step=0.5, key="finite_sigma")
    ec_block_size = st.select_slider("ECブロックサイズ [bit]", options=[256_000, 512_000, 1_000_000, 2_000_000, 5_000_000], value=DEFAULT_EC_BLOCK_SIZE, key="ec_block_size")
    qber_sample_bits = st.select_slider("QBER推定公開ビット/ECブロック", options=[0, 4096, 8192, 16384, 32768], value=DEFAULT_QBER_SAMPLE_BITS, key="qber_sample_bits")
    f_ec = st.slider("EC効率 fEC", 1.00, 1.50, DEFAULT_F_EC, step=0.01, key="f_ec")
    ec_fail_prob = st.slider("EC復号失敗率 [%]", 0.0, 5.0, DEFAULT_EC_FAIL_PERCENT, step=0.01, key="ec_fail_percent") / 100.0


def fmt_si(value, suffix="", decimals=2):
    x = float(value)
    sign = "-" if x < 0 else ""
    x = abs(x)
    for scale, unit in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if x >= scale:
            return f"{sign}{x / scale:.{decimals}f}{unit}{suffix}"
    if x.is_integer():
        return f"{sign}{int(x)}{suffix}"
    return f"{sign}{x:.{decimals}f}{suffix}"


def h2(q):
    q = min(max(float(q), 1e-12), 1 - 1e-12)
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)


def hash_preview(n_bits):
    if n_bits <= 0:
        return "-"
    preview_len = min(int(n_bits), 4096)
    seed = f"t12-2013-{n_bits}-{time.time()}"
    digest = hashlib.sha256(seed.encode()).digest()
    out = "".join(f"{b:08b}" for b in digest)
    while len(out) < preview_len:
        digest = hashlib.sha256((out + seed).encode()).digest()
        out += "".join(f"{b:08b}" for b in digest)
    return out[:preview_len]


def init_counts(protocol):
    counts = {
        "Protocol": protocol,
        "N": 0,
        "detected": 0,
        "key_len": 0,
        "key_errors": 0,
        "photon_0": 0,
        "photon_1": 0,
        "photon_multi": 0,
    }
    for basis in ["Z", "X"]:
        for lab in ["u", "v", "w"]:
            for prefix in ["N", "C", "E"]:
                counts[f"{prefix}_{lab}{basis}{basis}"] = 0
    return counts


def simulate_chunk(rng, size, protocol, counts):
    pz = 0.5 if protocol == "BB84参照" else P_Z_T12

    probs = np.array([P_U, P_V, P_W], dtype=float)
    probs = probs / probs.sum()
    intensity = rng.choice(np.array([0, 1, 2], dtype=np.int8), size=size, p=probs)
    mu = np.where(intensity == 0, MU_U, np.where(intensity == 1, MU_V, MU_W))

    alice_z = rng.random(size) < pz
    bob_z = rng.random(size) < pz
    basis_match = alice_z == bob_z

    photon_n = rng.poisson(mu)
    eta_i = 1.0 - np.power(1.0 - total_detection_efficiency, photon_n)
    photon_detected = rng.random(size) < eta_i
    dark_detected = rng.random(size) < dark_count_rate

    # Afterpulse handling for theory-oriented reproduction.
    # The paper reports APD afterpulse probability as a detector characteristic, but a proper
    # afterpulse model requires time correlation with previous avalanches. Treating every
    # afterpulse as an independent 50% random error overestimates QBER and can kill the key.
    # Therefore this simplified app uses afterpulse_percent only as an optional extra click
    # probability, and does not force a 50% error on those clicks.
    afterpulse_detected = (rng.random(size) < afterpulse_percent) & photon_detected & (~dark_detected)
    detected = photon_detected | dark_detected | afterpulse_detected

    # Error model: optical/modulation error plus dark-only random errors.
    # Afterpulse clicks inherit the ordinary optical error model in this simplified simulator.
    error_prob = np.full(size, optical_error_rate, dtype=np.float32)
    dark_only = dark_detected & (~photon_detected)
    error_prob[dark_only] = 0.5

    if eve_enabled:
        eve = (rng.random(size) < eve_rate) & detected
        eve_z = rng.random(size) < 0.5
        eve_wrong = eve & (eve_z != alice_z)
        error_prob[eve_wrong] = 0.5

    errors = rng.random(size) < error_prob
    matched_detected = basis_match & detected

    counts["N"] += size
    counts["detected"] += int(detected.sum())
    counts["photon_0"] += int((photon_n == 0).sum())
    counts["photon_1"] += int((photon_n == 1).sum())
    counts["photon_multi"] += int((photon_n >= 2).sum())

    # Track basis/intensity-specific sent pulses, counts and errors.
    for basis_name, basis_mask in [("Z", alice_z & bob_z), ("X", (~alice_z) & (~bob_z))]:
        for lab, code in [("u", 0), ("v", 1), ("w", 2)]:
            m_sent = (intensity == code) & basis_mask
            m_count = m_sent & detected
            counts[f"N_{lab}{basis_name}{basis_name}"] += int(m_sent.sum())
            counts[f"C_{lab}{basis_name}{basis_name}"] += int(m_count.sum())
            counts[f"E_{lab}{basis_name}{basis_name}"] += int((m_count & errors).sum())

    # Raw key candidates: signal pulses with matching bases, both bases contribute.
    signal = intensity == 0
    key_mask = signal & matched_detected
    counts["key_len"] += int(key_mask.sum())
    counts["key_errors"] += int((key_mask & errors).sum())


def simple_decoy_for_basis(c, basis):

    Nu = max(c[f"N_u{basis}{basis}"], 1)
    Nv = max(c[f"N_v{basis}{basis}"], 1)
    Nw = max(c[f"N_w{basis}{basis}"], 1)

    Cu = c[f"C_u{basis}{basis}"]
    Cv = c[f"C_v{basis}{basis}"]
    Cw = c[f"C_w{basis}{basis}"]

    Eu = c[f"E_u{basis}{basis}"]

    Qu = Cu / Nu
    Qv = Cv / Nv
    Qw = Cw / Nw

    Euu = Eu / max(Cu, 1)

    mu = MU_U
    nu = MU_V

    # vacuum yield
    Y0 = max(Qw, 1e-12)

    # ---------- Single photon yield ----------
    A = Qv * np.exp(nu)
    B = Qu * np.exp(mu) * (nu ** 2 / mu ** 2)
    C = ((mu ** 2 - nu ** 2) / mu ** 2) * Y0

    denom = mu * nu - nu ** 2

    if denom <= 0:
        Y1 = 0.0
    else:
        Y1 = mu / denom * (A - B - C)

    Y1 = np.clip(Y1, 1e-6, 1.0)

    # ---------- Single photon error ----------
    e1 = (
        Euu * Qu * np.exp(mu)
        - 0.5 * Y0
    ) / (mu * Y1)

    e1 = np.clip(e1, 0.0, 0.5)

    return {
        "Qu": Qu,
        "QBERu": Euu,
        "y0": Y0,
        "y1": Y1,
        "q1": e1,
        "N_u": Nu,
        "C_u": Cu,
    }

    return {
        "Qu": Qu,
        "QBERu": QBERu,
        "y0": y0,
        "y1": y1,
        "q1": e1,
        "N_u": Nu,
        "C_u": Cu,
    }


def calc_basis_key(c, key_basis, phase_basis, finite_sigma_value=None, ec_block_value=None, qber_sample_value=None):
    finite_sigma_value = finite_sigma if finite_sigma_value is None else finite_sigma_value
    ec_block_value = ec_block_size if ec_block_value is None else ec_block_value
    qber_sample_value = qber_sample_bits if qber_sample_value is None else qber_sample_value

    key_stats = simple_decoy_for_basis(c, key_basis)
    phase_stats = simple_decoy_for_basis(c, phase_basis)

    N_u = key_stats["N_u"]
    C_u = key_stats["C_u"]
    Qu = key_stats["Qu"]
    QBERu = key_stats["QBERu"]
    y0 = key_stats["y0"]
    y1 = key_stats["y1"]
    phase_counts = max(phase_stats["C_u"], 1)

    finite_margin = finite_sigma_value / np.sqrt(phase_counts)
    
    q1_phase = min(
        0.5,
        phase_stats["q1"] + finite_margin
    )

    # EC sample disclosure per block
    num_blocks = int(np.ceil(C_u / ec_block_value)) if C_u > 0 else 0
    sample_bits = min(C_u, num_blocks * qber_sample_value)

    # Paper Eq.(7)-like rate per N_u signal+basis pulses.

    Q1 = MU_U * np.exp(-MU_U) * y1

    S1 = Q1 * N_u
    S0 = np.exp(-MU_U) * y0 * N_u
    
    leakEC = C_u * f_ec * h2(QBERu)
    
    delta = finite_sigma_value * np.log2(max(C_u,2))
    
    secure_bits = max(
        0,
        int(
            S0
            + S1 * (1 - h2(q1_phase))
            - leakEC
            - delta
        )
    )
    
    secure_bits = max(
        0,
        secure_bits - sample_bits
    )
    
    secure_bits = int(
        secure_bits * (1 - ec_fail_prob)
    )
    
    return {
    "basis": key_basis,
    "N_u": N_u,
    "C_u": C_u,
    "QBERu": QBERu,
    "y0": y0,
    "y1": y1,
    "q1_phase": q1_phase,
    "num_blocks": num_blocks,
    "sample_bits": sample_bits,
    "secure_bits": secure_bits,
    }


def finalize_counts(c, finite_sigma_value=None, ec_block_value=None, qber_sample_value=None):
    z = calc_basis_key(c, "Z", "X", finite_sigma_value, ec_block_value, qber_sample_value)
    x = calc_basis_key(c, "X", "Z", finite_sigma_value, ec_block_value, qber_sample_value)
    final_key_len = z["secure_bits"] + x["secure_bits"]
    auth_cost = int(np.ceil(2 * np.log2(1 / EPS_TOTAL)))
    final_key_len = max(0, final_key_len - auth_cost)

    key_len = c["key_len"]
    qber = c["key_errors"] / max(key_len, 1)
    N = max(c["N"], 1)

    return {
        "Protocol": c["Protocol"],
        "送信パルス数": c["N"],
        "検出数": c["detected"],
        "鍵候補長": key_len,
        "QBER[%]": qber * 100,
        "最終鍵長": final_key_len,
        "推定secure rate[Mb/s]": PULSE_RATE_HZ * final_key_len / N / 1e6,
        "推定sifted rate[Mb/s]": PULSE_RATE_HZ * key_len / N / 1e6,
        "Z安全鍵長": z["secure_bits"],
        "X安全鍵長": x["secure_bits"],
        "Z_QBER[%]": z["QBERu"] * 100,
        "X_QBER[%]": x["QBERu"] * 100,
        "Z_y0": z["y0"],
        "Z_y1": z["y1"],
        "Z_q1_phase[%]": z["q1_phase"] * 100,
        "X_y0": x["y0"],
        "X_y1": x["y1"],
        "X_q1_phase[%]": x["q1_phase"] * 100,
        "ECブロック数": z["num_blocks"] + x["num_blocks"],
        "QBER推定公開ビット": z["sample_bits"] + x["sample_bits"],
        "0光子率[%]": c["photon_0"] / N * 100,
        "1光子率[%]": c["photon_1"] / N * 100,
        "多光子率[%]": c["photon_multi"] / N * 100,
        "final_key": hash_preview(final_key_len),
        "can_generate_key": final_key_len > 0 and qber * 100 <= qber_threshold,
        "_counts": c,
    }


def run_stream(protocol, progress_bar=None, status_box=None):
    counts = init_counts(protocol)
    rng = np.random.default_rng(int(time.time() * 1000) % 2**32 + (0 if protocol == "BB84参照" else 12345))
    remaining = dataset_size
    done = 0
    while remaining > 0:
        m = min(chunk_size, remaining)
        simulate_chunk(rng, m, protocol, counts)
        remaining -= m
        done += m
        if progress_bar is not None:
            progress_bar.progress(done / dataset_size)
        if status_box is not None:
            status_box.caption(f"{protocol}: {fmt_si(done)} / {fmt_si(dataset_size)} パルス処理済み")
    return finalize_counts(counts)


def metric_cards(s):
    items = [
        ("送信", fmt_si(s["送信パルス数"])),
        ("検出", fmt_si(s["検出数"])),
        ("鍵候補", fmt_si(s["鍵候補長"])),
        ("QBER", f"{s['QBER[%]']:.2f}%"),
        ("最終鍵", fmt_si(s["最終鍵長"], "bit")),
        ("推定SKR", fmt_si(s["推定secure rate[Mb/s]"] * 1e6, "b/s")),
    ]
    row1 = st.columns(3)
    row2 = st.columns(3)
    for col, (label, value) in zip(row1, items[:3]):
        with col:
            st.markdown(f"**{label}**")
            st.markdown(f"### `{value}`")
    for col, (label, value) in zip(row2, items[3:]):
        with col:
            st.markdown(f"**{label}**")
            st.markdown(f"### `{value}`")


def format_df(df):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if "[%]" in col or "QBER" in col or "q1" in col:
                out[col] = out[col].map(lambda x: f"{x:.3f}")
            elif "rate" in col or "Mb/s" in col:
                out[col] = out[col].map(lambda x: f"{x:.3f}")
            elif "y" in col or "Q" == col:
                out[col] = out[col].map(lambda x: f"{x:.4e}")
            else:
                out[col] = out[col].map(lambda x: fmt_si(x))
    return out


def horizontal_bar(df, cols, title, x_title):
    plot_df = df.melt(id_vars="プロトコル", value_vars=cols, var_name="項目", value_name="値")
    fig = px.bar(plot_df, y="プロトコル", x="値", color="項目", orientation="h", barmode="group", text="値", title=title)
    fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=x_title, yaxis_title="", legend_title="", height=360, margin=dict(l=40, r=100, t=60, b=40))
    return fig


if st.button("2013論文再現シミュレーション実行", type="primary"):
    st.warning("100.66Mパルスでは数十秒程度かかる場合があります。")
    st.dataframe(BENCHMARKS, use_container_width=True)
    start = time.time()
    summaries = []

    if protocol_mode in ["比較表示", "BB84参照のみ"]:
        st.subheader("BB84参照を計算中")
        pb = st.progress(0)
        box = st.empty()
        summaries.append(run_stream("BB84参照", pb, box))

    if protocol_mode in ["比較表示", "T12論文値のみ"]:
        st.subheader("T12論文値を計算中")
        pb = st.progress(0)
        box = st.empty()
        summaries.append(run_stream("T12論文値", pb, box))

    st.success(f"計算完了: {time.time() - start:.1f} 秒")

    st.subheader("1. 全体結果")
    for s in summaries:
        st.markdown(f"#### {s['Protocol']}")
        metric_cards(s)

    st.subheader("2. 基底別安全鍵寄与")
    basis_df = pd.DataFrame([
        {
            "プロトコル": s["Protocol"],
            "Z安全鍵長": s["Z安全鍵長"],
            "X安全鍵長": s["X安全鍵長"],
            "Z_QBER[%]": s["Z_QBER[%]"],
            "X_QBER[%]": s["X_QBER[%]"],
            "Z_y1": s["Z_y1"],
            "X_y1": s["X_y1"],
            "Z_q1_phase[%]": s["Z_q1_phase[%]"],
            "X_q1_phase[%]": s["X_q1_phase[%]"],
        }
        for s in summaries
    ])
    st.dataframe(format_df(basis_df), use_container_width=True)

    st.subheader("3. BB84 ⇔ T12 比較")
    compare_df = pd.DataFrame([
        {
            "プロトコル": s["Protocol"],
            "推定sifted rate[Mb/s]": s["推定sifted rate[Mb/s]"],
            "推定secure rate[Mb/s]": s["推定secure rate[Mb/s]"],
        }
        for s in summaries
    ])
    st.plotly_chart(horizontal_bar(compare_df, ["推定sifted rate[Mb/s]", "推定secure rate[Mb/s]"], "Sifted / Secure rate 比較", "Mb/s"), use_container_width=True)
    st.dataframe(format_df(compare_df), use_container_width=True)

    st.subheader("4. 論文ベンチマークとの比較")
    bench50 = BENCHMARKS[BENCHMARKS["距離[km]"] == 50].iloc[0]
    bench_rows = []
    for s in summaries:
        target = bench50["T12 SKR[Mb/s]"] if s["Protocol"] == "T12論文値" else bench50["BB84 SKR[Mb/s]"]
        bench_rows.append({
            "プロトコル": s["Protocol"],
            "今回SKR[Mb/s]": s["推定secure rate[Mb/s]"],
            "50km論文値[Mb/s]": target,
            "差分[Mb/s]": s["推定secure rate[Mb/s]"] - target,
        })
    bench_df = pd.DataFrame(bench_rows)
    st.dataframe(format_df(bench_df), use_container_width=True)

    st.subheader("5. 光子数分布")
    photon_df = pd.DataFrame([
        {"プロトコル": s["Protocol"], "分類": "0光子", "割合[%]": s["0光子率[%]"]} for s in summaries
    ] + [
        {"プロトコル": s["Protocol"], "分類": "1光子", "割合[%]": s["1光子率[%]"]} for s in summaries
    ] + [
        {"プロトコル": s["Protocol"], "分類": "多光子", "割合[%]": s["多光子率[%]"]} for s in summaries
    ])
    fig_photon = px.bar(photon_df, y="プロトコル", x="割合[%]", color="分類", orientation="h", barmode="group", text="割合[%]", title="光子数分布")
    fig_photon.update_traces(texttemplate="%{x:.2f}%", textposition="outside", cliponaxis=False)
    fig_photon.update_layout(xaxis_title="割合[%]", yaxis_title="", legend_title="", height=360, margin=dict(l=40, r=100, t=60, b=40))
    st.plotly_chart(fig_photon, use_container_width=True)

    target = next((s for s in summaries if s["Protocol"] == "T12論文値"), summaries[0])
    counts = target["_counts"]

    st.subheader("6. ECブロックサイズ変更時のSKR変化")
    block_rows = []
    for block in [256_000, 512_000, 1_000_000, 2_000_000, 5_000_000]:
        ss = finalize_counts(counts, ec_block_value=block)
        block_rows.append({"ECブロックサイズ": block, "SKR[Mb/s]": ss["推定secure rate[Mb/s]"]})
    block_df = pd.DataFrame(block_rows)
    st.plotly_chart(px.line(block_df, x="ECブロックサイズ", y="SKR[Mb/s]", markers=True, title="ECブロックサイズとSKR"), use_container_width=True)
    st.dataframe(format_df(block_df), use_container_width=True)

    st.subheader("7. finite_sigma変更時のSKR変化")
    sigma_rows = []
    for sig in [0, 0.5, 1, 2, 3, 5, 8, 10]:
        ss = finalize_counts(counts, finite_sigma_value=sig)
        sigma_rows.append({"finite_sigma": sig, "SKR[Mb/s]": ss["推定secure rate[Mb/s]"]})
    sigma_df = pd.DataFrame(sigma_rows)
    st.plotly_chart(px.line(sigma_df, x="finite_sigma", y="SKR[Mb/s]", markers=True, title="finite_sigmaとSKR"), use_container_width=True)
    st.dataframe(format_df(sigma_df), use_container_width=True)

    st.subheader("8. 最終鍵プレビュー")
    for s in summaries:
        with st.expander(f"{s['Protocol']} の最終鍵プレビュー", expanded=False):
            if s["can_generate_key"]:
                st.caption("表示は先頭最大4096 bitのプレビューです。実際の最終鍵長は上のメトリクスを参照してください。")
                st.code(s["final_key"], language="text")
            else:
                st.error("QBERが高い、または後処理後の鍵長が0以下のため、最終鍵は生成されませんでした。")
else:
    st.info("左の条件を設定して、［2013論文再現シミュレーション実行］を押してください。")
