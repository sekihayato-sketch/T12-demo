import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
import hashlib

st.set_page_config(page_title="T12 100.66Mb Streaming Simulator", layout="wide")

# =============================
# T12 paper-like constants
# =============================
PULSE_RATE_HZ = 1_000_000_000
PA_DATASET_BITS_DEFAULT = 100.66e6
MU_U = 0.4
MU_V = 0.1
MU_W = 0.0007
P_U = 0.96973
P_V = 0.01661
P_W = 0.01466
P_Z_T12 = 0.96677
P_ST = 1 / 128
EPS_AUTH = 1e-10
EC_FAIL_PROB_DEFAULT = 0.0073

st.title("BB84 / T12 100.66Mb 実装値寄せシミュレータ")
st.caption("基底表・詳細表・アニメーションを削除し、100.66Mb級データセットをチャンク処理で直接シミュレーションします。")

st.markdown("""
この版は、**100.66Mb PA datasetに近づけることを優先**した軽量版です。
巨大なビット列・詳細表・アニメーションは保持せず、チャンクごとに集計値だけを加算します。

- T12論文値：`pZ=96.677%`, `pX=3.323%`, `u/v/w=0.4/0.1/0.0007`, `p_u/p_v/p_w=96.973%/1.661%/1.466%`
- 実際に **100.66Mパルス級** を直接シミュレーション可能
- 表示は `K/M/G/T` 単位に整形
- 詳細表・アニメーションは削除
""")

with st.sidebar:
    st.header("シミュレーション条件")
    dataset_size = st.select_slider(
        "送信パルス数 / PA datasetサイズ",
        options=[1_048_576, 4_194_304, 16_777_216, 33_554_432, 67_108_864, 100_660_000],
        value=100_660_000,
    )
    protocol_mode = st.radio("表示モード", ["比較表示", "BB84参照のみ", "T12論文値のみ"], index=0)
    chunk_size = st.select_slider(
        "内部処理チャンクサイズ",
        options=[250_000, 500_000, 1_000_000, 2_000_000, 5_000_000],
        value=1_000_000,
    )
    st.caption("チャンクサイズを大きくすると速くなる場合がありますが、メモリ使用量も増えます。")

    st.markdown("---")
    st.subheader("物理・検出条件")
    total_detection_efficiency = st.slider("総合検出効率 η [%]", 1.0, 50.0, 13.5, step=0.1) / 100.0
    dark_count_rate = st.slider("暗計数率 Y0 [%/pulse]", 0.0, 1.0, 0.045, step=0.001) / 100.0
    optical_error_rate = st.slider("光学系・通信路由来の誤り率 Eopt [%]", 0.0, 10.0, 3.0, step=0.1) / 100.0
    afterpulse_error_rate = st.slider("アフターパルス由来の追加誤り率 [%]", 0.0, 10.0, 0.0, step=0.1) / 100.0

    st.markdown("---")
    st.subheader("Eve")
    eve_enabled = st.checkbox("遮断・再送信攻撃を有効化", value=False)
    eve_rate = st.slider("Eve介入率 [%]", 0, 100, 0, step=5, disabled=not eve_enabled) / 100.0

    st.markdown("---")
    st.subheader("後処理")
    qber_threshold = st.slider("鍵破棄しきい値 QBER [%]", 0.0, 20.0, 11.0, step=0.5)
    finite_sigma = st.slider("有限サイズ補正 sigma", 0.0, 10.0, 1.0, step=0.5)
    ec_fail_prob = st.slider("EC復号失敗率 [%]", 0.0, 5.0, EC_FAIL_PROB_DEFAULT * 100, step=0.01) / 100.0
    ec_model = st.radio("ECモデル", ["LDPC風・QBER依存", "固定fEC"], index=0)
    fixed_f_ec = st.slider("固定 fEC", 1.00, 2.00, 1.34, step=0.01)


def fmt_si(value, suffix="", decimals=2):
    try:
        x = float(value)
    except Exception:
        return str(value)
    sign = "-" if x < 0 else ""
    x = abs(x)
    for scale, unit in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if x >= scale:
            return f"{sign}{x/scale:.{decimals}f}{unit}{suffix}"
    if x.is_integer():
        return f"{sign}{int(x)}{suffix}"
    return f"{sign}{x:.{decimals}f}{suffix}"


def fmt_pct(x, decimals=2):
    return f"{x:.{decimals}f}%"


def h2(q):
    q = min(max(float(q), 1e-12), 1 - 1e-12)
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)


def f_ec_ldpc_like(qber):
    q = qber * 100
    if q <= 2:
        return 1.20
    if q <= 3:
        return 1.20 + (q - 2) * (1.34 - 1.20)
    if q <= 5:
        return 1.34 + (q - 3) * (1.38 - 1.34) / 2
    if q <= 10:
        return 1.38 + (q - 5) * (1.50 - 1.38) / 5
    return 1.60


def hash_preview(n_bits):
    if n_bits <= 0:
        return "-"
    preview_len = min(int(n_bits), 4096)
    seed_text = f"T12-preview-{n_bits}-{time.time()}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    out = "".join(f"{b:08b}" for b in digest)
    while len(out) < preview_len:
        digest = hashlib.sha256((out + seed_text).encode("utf-8")).digest()
        out += "".join(f"{b:08b}" for b in digest)
    return out[:preview_len]


def init_counts(protocol):
    return {
        "Protocol": protocol,
        "N": 0,
        "stabilization": 0,
        "u": 0,
        "v": 0,
        "w": 0,
        "detected": 0,
        "basis_match": 0,
        "z_match": 0,
        "x_match": 0,
        "key_len": 0,
        "key_errors": 0,
        "x_len": 0,
        "x_errors": 0,
        "N_u": 0,
        "D_u": 0,
        "Err_u": 0,
        "N_v": 0,
        "D_v": 0,
        "Err_v": 0,
        "N_w": 0,
        "D_w": 0,
        "Err_w": 0,
        "photon_0": 0,
        "photon_1": 0,
        "photon_multi": 0,
    }


def simulate_chunk(rng, size, protocol, counts):
    pz = 0.5 if protocol == "BB84参照" else P_Z_T12

    # Stabilization and intensity choice
    stabilization = rng.random(size) < P_ST
    probs = np.array([P_U, P_V, P_W], dtype=float)
    probs = probs / probs.sum()
    intensity = rng.choice(np.array([0, 1, 2], dtype=np.int8), size=size, p=probs)
    mu = np.where(intensity == 0, MU_U, np.where(intensity == 1, MU_V, MU_W))

    alice_bits = rng.integers(0, 2, size=size, dtype=np.int8)
    alice_z = rng.random(size) < pz
    bob_z = rng.random(size) < pz

    photon_n = rng.poisson(mu)
    eta_i = 1.0 - np.power(1.0 - total_detection_efficiency, photon_n)
    photon_detected = rng.random(size) < eta_i
    dark_detected = rng.random(size) < dark_count_rate
    detected = (photon_detected | dark_detected) & (~stabilization)

    # Basis match
    basis_match = detected & (alice_z == bob_z)
    z_match = basis_match & alice_z
    x_match = basis_match & (~alice_z)
    signal = intensity == 0

    if protocol == "T12論文値":
        key_mask = signal & z_match
        check_mask = x_match
    else:
        key_mask = signal & basis_match
        check_mask = x_match

    # Error model
    base_error = min(1.0, optical_error_rate + afterpulse_error_rate)
    error_prob = np.full(size, base_error, dtype=np.float32)
    dark_only = dark_detected & (~photon_detected) & detected
    error_prob[dark_only] = 0.5

    if eve_enabled:
        eve = (rng.random(size) < eve_rate) & detected
        eve_z = rng.random(size) < 0.5
        eve_wrong = eve & (eve_z != alice_z)
        error_prob[eve_wrong] = 0.5
    else:
        eve = np.zeros(size, dtype=bool)

    errors = rng.random(size) < error_prob

    key_errors = key_mask & errors
    x_errors = check_mask & errors

    # counts
    counts["N"] += size
    counts["stabilization"] += int(stabilization.sum())
    counts["u"] += int((intensity == 0).sum())
    counts["v"] += int((intensity == 1).sum())
    counts["w"] += int((intensity == 2).sum())
    counts["detected"] += int(detected.sum())
    counts["basis_match"] += int(basis_match.sum())
    counts["z_match"] += int(z_match.sum())
    counts["x_match"] += int(x_match.sum())
    counts["key_len"] += int(key_mask.sum())
    counts["key_errors"] += int(key_errors.sum())
    counts["x_len"] += int(check_mask.sum())
    counts["x_errors"] += int(x_errors.sum())
    counts["photon_0"] += int((photon_n == 0).sum())
    counts["photon_1"] += int((photon_n == 1).sum())
    counts["photon_multi"] += int((photon_n >= 2).sum())

    non_stab = ~stabilization
    for label, code in [("u", 0), ("v", 1), ("w", 2)]:
        m_int = intensity == code
        m_basis = basis_match & m_int
        counts[f"N_{label}"] += int((m_int & non_stab).sum())
        counts[f"D_{label}"] += int(m_basis.sum())
        counts[f"Err_{label}"] += int((m_basis & errors).sum())


def decoy_estimate(c):
    N_u = max(c["N_u"], 1)
    N_v = max(c["N_v"], 1)
    N_w = max(c["N_w"], 1)
    Q_u = c["D_u"] / N_u
    Q_v = c["D_v"] / N_v
    Q_w = c["D_w"] / N_w
    E_v = c["Err_v"] / max(c["D_v"], 1)

    mu = MU_U
    nu = MU_V
    Y0 = Q_w
    denom = mu * nu - nu * nu
    if denom <= 0:
        Y1_L = 0.0
    else:
        Y1_L = (mu / denom) * (
            Q_v * np.exp(nu)
            - Q_u * np.exp(mu) * (nu * nu / (mu * mu))
            - ((mu * mu - nu * nu) / (mu * mu)) * Y0
        )
        Y1_L = max(0.0, min(1.0, Y1_L))

    Q1_L = mu * np.exp(-mu) * Y1_L
    if nu * Y1_L > 0:
        e1_U = (E_v * Q_v * np.exp(nu) - 0.5 * Y0) / (nu * Y1_L)
        e1_U = max(0.0, min(0.5, e1_U))
    else:
        e1_U = 0.5

    return {"Q_u": Q_u, "Q_v": Q_v, "Q_w": Q_w, "Y0_est": Y0, "Y1_L": Y1_L, "Q1_L": Q1_L, "e1_U": e1_U}


def finalize_counts(c):
    N = max(c["N"], 1)
    key_len = c["key_len"]
    qber = c["key_errors"] / max(key_len, 1)
    x_qber = c["x_errors"] / max(c["x_len"], 1)
    finite_margin = 0.2 * finite_sigma / np.sqrt(max(c["x_len"], 1))
    phase_error = min(0.5, x_qber + finite_margin)
    f_ec = f_ec_ldpc_like(qber) if ec_model == "LDPC風・QBER依存" else fixed_f_ec
    ec_leakage = int(round(f_ec * h2(qber) * key_len)) if key_len > 0 else 0
    privacy_term = int(round(key_len * h2(phase_error))) if key_len > 0 else 0
    finite_penalty = int(np.ceil(finite_sigma * np.sqrt(max(key_len, 1)) * np.log2(max(key_len, 2)))) if key_len > 0 else 0
    auth_cost = int(np.ceil(2 * np.log2(1 / EPS_AUTH)))
    raw_secure = key_len - ec_leakage - privacy_term - finite_penalty - auth_cost
    final_key_len = int(max(0, raw_secure) * (1 - ec_fail_prob))
    can_generate = key_len > 0 and qber * 100 <= qber_threshold and final_key_len > 0
    if not can_generate:
        final_key_len = 0

    decoy = decoy_estimate(c)
    if c["Protocol"] == "T12論文値":
        theory_sift = (1 - P_ST) * P_U * P_Z_T12 * P_Z_T12
    else:
        theory_sift = (1 - P_ST) * P_U * 0.5

    return {
        "Protocol": c["Protocol"],
        "送信パルス数": c["N"],
        "検出数": c["detected"],
        "基底一致数": c["basis_match"],
        "Z/Z一致数": c["z_match"],
        "X/X一致数": c["x_match"],
        "鍵候補長": key_len,
        "鍵候補効率[%]": key_len / N * 100,
        "理論選別効率[%]": theory_sift * 100,
        "誤り数": c["key_errors"],
        "QBER[%]": qber * 100,
        "X基底QBER[%]": x_qber * 100,
        "phase error[%]": phase_error * 100,
        "fEC": f_ec,
        "EC leakage[bit]": ec_leakage,
        "privacy term[bit]": privacy_term,
        "finite penalty[bit]": finite_penalty,
        "auth cost[bit]": auth_cost,
        "EC fail prob[%]": ec_fail_prob * 100,
        "最終鍵長": final_key_len,
        "最終鍵効率[%]": final_key_len / N * 100,
        "推定sifted rate[Mb/s]": PULSE_RATE_HZ * key_len / N / 1e6,
        "推定secure rate[Mb/s]": PULSE_RATE_HZ * final_key_len / N / 1e6,
        "0光子率[%]": c["photon_0"] / N * 100,
        "1光子率[%]": c["photon_1"] / N * 100,
        "多光子率[%]": c["photon_multi"] / N * 100,
        "Y0_est": decoy["Y0_est"],
        "Y1_L": decoy["Y1_L"],
        "Q1_L": decoy["Q1_L"],
        "e1_U[%]": decoy["e1_U"] * 100,
        "final_key": hash_preview(final_key_len),
        "can_generate_key": can_generate,
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
        ("QBER", fmt_pct(s["QBER[%]"])),
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

def format_numeric_tables(df):
    out = df.copy()
    for col in out.columns:
        if col in ["プロトコル"]:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            if "[%]" in col or "QBER" in col or "error" in col or "効率" in col or "prob" in col:
                out[col] = out[col].map(lambda x: f"{x:.3f}")
            elif "rate" in col or "Mb/s" in col:
                out[col] = out[col].map(lambda x: f"{x:.2f}")
            else:
                out[col] = out[col].map(lambda x: fmt_si(x))
    return out


def plot_horizontal_grouped(df, value_cols, title, x_title):
    plot_df = df.melt(id_vars="プロトコル", value_vars=value_cols, var_name="項目", value_name="値")
    fig = px.bar(
        plot_df,
        y="プロトコル",
        x="値",
        color="項目",
        orientation="h",
        barmode="group",
        text="値",
        title=title,
    )
    fig.update_traces(texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="",
        legend_title="",
        height=360,
        margin=dict(l=40, r=80, t=60, b=40),
        font=dict(size=14),
    )
    return fig


if st.button("100.66Mb級シミュレーション実行", type="primary"):
    st.warning("100.66Mパルスでは数十秒〜数分かかる場合があります。Streamlit Cloudの負荷が高い場合は、まず16.78Mまたは67.11Mで確認してください。")
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

    elapsed = time.time() - start
    st.success(f"計算完了: {elapsed:.1f} 秒")

    st.subheader("1. 全体結果")
    for s in summaries:
        st.markdown(f"#### {s['Protocol']}")
        metric_cards(s)

    st.subheader("2. 後処理内訳")
    post_df = pd.DataFrame([
        {
            "プロトコル": s["Protocol"],
            "鍵候補長": s["鍵候補長"],
            "QBER[%]": s["QBER[%]"],
            "phase error[%]": s["phase error[%]"],
            "fEC": s["fEC"],
            "EC leakage[bit]": s["EC leakage[bit]"],
            "privacy term[bit]": s["privacy term[bit]"],
            "finite penalty[bit]": s["finite penalty[bit]"],
            "auth cost[bit]": s["auth cost[bit]"],
            "EC fail prob[%]": s["EC fail prob[%]"],
            "最終鍵長": s["最終鍵長"],
            "推定secure rate[Mb/s]": s["推定secure rate[Mb/s]"],
        }
        for s in summaries
    ])
    st.dataframe(format_numeric_tables(post_df), use_container_width=True)

    st.subheader("3. 効率比較")
    compare_df = pd.DataFrame([
        {
            "プロトコル": s["Protocol"],
            "理論選別効率[%]": s["理論選別効率[%]"],
            "鍵候補効率[%]": s["鍵候補効率[%]"],
            "最終鍵効率[%]": s["最終鍵効率[%]"],
            "推定sifted rate[Mb/s]": s["推定sifted rate[Mb/s]"],
            "推定secure rate[Mb/s]": s["推定secure rate[Mb/s]"],
        }
        for s in summaries
    ])

    st.markdown("#### レート比較")
    fig_rate = plot_horizontal_grouped(
        compare_df,
        ["推定sifted rate[Mb/s]", "推定secure rate[Mb/s]"],
        "Sifted rate / Secure rate 比較",
        "Mb/s",
    )
    st.plotly_chart(fig_rate, use_container_width=True)

    st.markdown("#### 効率比較")
    fig_eff = plot_horizontal_grouped(
        compare_df,
        ["理論選別効率[%]", "鍵候補効率[%]", "最終鍵効率[%]"],
        "選別効率 / 鍵候補効率 / 最終鍵効率",
        "%",
    )
    st.plotly_chart(fig_eff, use_container_width=True)
    st.dataframe(format_numeric_tables(compare_df), use_container_width=True)

    st.subheader("4. デコイ解析・光子数分布")
    decoy_df = pd.DataFrame([
        {
            "プロトコル": s["Protocol"],
            "Y0_est": s["Y0_est"],
            "Y1_L": s["Y1_L"],
            "Q1_L": s["Q1_L"],
            "e1_U[%]": s["e1_U[%]"],
            "X基底QBER[%]": s["X基底QBER[%]"],
            "phase error[%]": s["phase error[%]"],
            "0光子率[%]": s["0光子率[%]"],
            "1光子率[%]": s["1光子率[%]"],
            "多光子率[%]": s["多光子率[%]"],
        }
        for s in summaries
    ])

    photon_df = decoy_df[["プロトコル", "0光子率[%]", "1光子率[%]", "多光子率[%]"]].melt(
        id_vars="プロトコル", var_name="光子数分類", value_name="割合[%]"
    )
    fig_photon = px.bar(
        photon_df,
        y="プロトコル",
        x="割合[%]",
        color="光子数分類",
        orientation="h",
        barmode="group",
        text="割合[%]",
        title="光子数分布",
    )
    fig_photon.update_traces(texttemplate="%{x:.2f}%", textposition="outside", cliponaxis=False)
    fig_photon.update_layout(
        xaxis_title="割合[%]",
        yaxis_title="",
        legend_title="",
        height=360,
        margin=dict(l=40, r=100, t=60, b=40),
        font=dict(size=14),
    )
    st.plotly_chart(fig_photon, use_container_width=True)

    st.markdown("#### デコイ推定値")
    st.dataframe(format_numeric_tables(decoy_df), use_container_width=True)

    st.subheader("5. 最終鍵プレビュー")
    for s in summaries:
        with st.expander(f"{s['Protocol']} の最終鍵プレビュー", expanded=False):
            if s["can_generate_key"] and s["最終鍵長"] > 0:
                st.caption("表示は先頭最大4096 bitのプレビューです。実際の最終鍵長は上のメトリクスを参照してください。")
                st.code(s["final_key"], language="text")
            else:
                st.error("QBERが高い、または後処理後の鍵長が0以下のため、最終鍵は生成されませんでした。")

else:
    st.info("左の条件を設定して、［100.66Mb級シミュレーション実行］を押してください。")
