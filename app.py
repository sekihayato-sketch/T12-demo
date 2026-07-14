import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import time
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(page_title="BB84 / T12 QKD Simulator", layout="wide")

st.title("BB84 / T12 論文値反映 QKDシミュレータ")
st.caption("T12論文パラメータ：pZ=96.677%, pX=3.323%, u/v/w=0.4/0.1/0.0007, p_u/p_v/p_w=96.973%/1.661%/1.466%")

st.markdown("""
このアプリでは、従来BB84参照モデルと、T12論文値を反映したモデルを比較します。

- **BB84参照モデル**：基底選択を Z/X = 50/50 とした比較用モデル
- **T12論文値モデル**：基底選択を **Z=96.677%, X=3.323%** とし、信号/デコイ/真空強度を論文値で選択
- 光子数は、各パルスで選ばれた平均光子数 μ に対して **Poisson(μ)** で生成します。

※ このアプリは教育・理解用の簡略シミュレータです。有限サイズ解析、デコイ推定、LDPC/PAの厳密実装そのものではありません。
""")

# -----------------------------
# Paper default values
# -----------------------------
DEFAULT_PULSE_RATE_HZ = 1_000_000_000
DEFAULT_MU_U = 0.4
DEFAULT_MU_V = 0.1
DEFAULT_MU_W = 0.0007
DEFAULT_P_U = 0.96973
DEFAULT_P_V = 0.01661
DEFAULT_P_W = 0.01466
DEFAULT_P_Z = 0.96677
DEFAULT_P_X = 0.03323
DEFAULT_P_ST = 1 / 128
DEFAULT_PA_DATASET_BITS = 100.66e6
DEFAULT_EPSILON = 1e-10
DEFAULT_PA_COMPRESSION = 0.29
DEFAULT_F_EC = 1.34

with st.expander("実装したT12論文値", expanded=True):
    st.markdown(f"""
    ### T12論文値として反映している値

    ```text
    パルス繰り返しレート f       = 1 GHz
    強度 u, v, w                = 0.4, 0.1, 0.0007 photons/pulse
    強度選択確率 p_u,p_v,p_w    = 96.973%, 1.661%, 1.466%
    基底選択確率 p_Z,p_X        = 96.677%, 3.323%
    安定化スロット p_st         = 1/128 = {DEFAULT_P_ST:.6f}
    PA dataset                  = 100.66 Mb
    security parameter epsilon  = 1e-10
    PA圧縮比の代表値            = 約0.29
    ```

    論文中の選別効率は、概念的に以下で確認できます。

    ```text
    eta_sift = (1 - p_st) * p_u * p_Z^2
             = (1 - 1/128) * 0.96973 * 0.96677^2
             ≒ 0.899
    ```
    """)

with st.sidebar:
    st.header("シミュレーション条件")

    num_pulses = st.select_slider(
        "送信パルス数",
        options=[32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576],
        value=4096,
    )

    protocol_mode = st.radio("表示モード", ["比較表示", "BB84参照のみ", "T12論文値のみ"], index=0)

    st.markdown("---")
    st.subheader("T12論文値")
    use_paper_values = st.checkbox("T12論文値を固定して使う", value=True)

    if use_paper_values:
        p_z_t12 = DEFAULT_P_Z
        p_x_t12 = DEFAULT_P_X
        mu_u = DEFAULT_MU_U
        mu_v = DEFAULT_MU_V
        mu_w = DEFAULT_MU_W
        p_u = DEFAULT_P_U
        p_v = DEFAULT_P_V
        p_w = DEFAULT_P_W
        p_st = DEFAULT_P_ST
        pa_compression = DEFAULT_PA_COMPRESSION
        f_ec = DEFAULT_F_EC
    else:
        p_z_t12 = st.slider("T12 pZ [%]", 50.0, 99.9, DEFAULT_P_Z * 100, step=0.001) / 100.0
        p_x_t12 = 1.0 - p_z_t12
        mu_u = st.slider("signal u", 0.001, 2.0, DEFAULT_MU_U, step=0.001)
        mu_v = st.slider("decoy v", 0.001, 1.0, DEFAULT_MU_V, step=0.001)
        mu_w = st.slider("vacuum w", 0.0, 0.01, DEFAULT_MU_W, step=0.0001)
        p_u = st.slider("p_u [%]", 0.0, 100.0, DEFAULT_P_U * 100, step=0.001) / 100.0
        p_v = st.slider("p_v [%]", 0.0, 100.0, DEFAULT_P_V * 100, step=0.001) / 100.0
        p_w = max(0.0, 1.0 - p_u - p_v)
        p_st = st.slider("p_st [%]", 0.0, 5.0, DEFAULT_P_ST * 100, step=0.001) / 100.0
        pa_compression = st.slider("PA圧縮比", 0.01, 1.00, DEFAULT_PA_COMPRESSION, step=0.01)
        f_ec = st.slider("EC効率 fEC", 1.00, 2.00, DEFAULT_F_EC, step=0.01)

    st.caption(f"現在値: pZ={p_z_t12*100:.3f}%, pX={p_x_t12*100:.3f}%, p_u={p_u*100:.3f}%, p_v={p_v*100:.3f}%, p_w={p_w*100:.3f}%")

    st.markdown("---")
    st.subheader("実験条件")
    total_detection_efficiency = st.slider("総合検出効率 η [%]", 1.0, 100.0, 20.0, step=1.0) / 100.0
    dark_count_rate = st.slider("暗計数率 Y0 [%/pulse]", 0.0, 5.0, 0.0, step=0.1) / 100.0
    optical_error_rate = st.slider("光学系・通信路由来の誤り率 Eopt [%]", 0.0, 20.0, 3.0, step=0.1) / 100.0

    eve_enabled = st.checkbox("Eveによる遮断・再送信攻撃を有効化", value=False)
    eve_rate = st.slider("Eve介入率 [%]", 0, 100, 0, step=5, disabled=not eve_enabled) / 100.0

    qber_threshold = st.slider("鍵破棄しきい値 QBER [%]", 0.0, 20.0, 11.0, step=0.5)
    show_bit_motion = st.checkbox("0/1送信アニメーションを表示", value=True)
    show_detail_table = st.checkbox("詳細表を表示", value=True)
    max_display_rows = st.slider("詳細表の最大行数", 50, 1000, 500, step=50)


def h2(q):
    q = min(max(q, 1e-12), 1 - 1e-12)
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)


def bits_to_string(bits):
    if len(bits) == 0:
        return "-"
    return "".join(str(int(b)) for b in bits)


def hash_to_bits(text, length):
    if length <= 0:
        return "-"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    bit_string = "".join(f"{byte:08b}" for byte in digest)
    while len(bit_string) < length:
        digest = hashlib.sha256((bit_string + text).encode("utf-8")).digest()
        bit_string += "".join(f"{byte:08b}" for byte in digest)
    return bit_string[:length]


def choose_bases(rng, n, protocol):
    if protocol == "BB84参照":
        pz = 0.5
    else:
        pz = p_z_t12
    px = 1.0 - pz
    alice = rng.choice(["Z", "X"], size=n, p=[pz, px])
    bob = rng.choice(["Z", "X"], size=n, p=[pz, px])
    return alice, bob


def choose_intensities(rng, n):

    probs = np.array([p_u, p_v, p_w], dtype=float)

    probs /= probs.sum()

    labels = rng.choice(
        ["u", "v", "w"],
        size=n,
        p=probs
    )

    mu = np.where(
        labels == "u",
        mu_u,
        np.where(labels == "v", mu_v, mu_w)
    )

    return labels, mu


def simulate_protocol(protocol, n, seed):
    rng = np.random.default_rng(seed)

    stabilization_slot = rng.random(n) < p_st
    intensity_label, mu_values = choose_intensities(rng, n)

    alice_bits = rng.integers(0, 2, size=n)
    alice_bases, bob_bases = choose_bases(rng, n, protocol)

    photon_numbers = rng.poisson(mu_values)
    eta_i = 1.0 - np.power(1.0 - total_detection_efficiency, photon_numbers)
    photon_detected = rng.random(n) < eta_i
    dark_detected = rng.random(n) < dark_count_rate
    detected = (photon_detected | dark_detected) & (~stabilization_slot)

    # Eve intercept-resend, simplified
    eve_intervenes = eve_enabled & (rng.random(n) < eve_rate) & detected
    eve_bases = np.full(n, "-", dtype=object)
    eve_results = np.full(n, -1, dtype=int)
    transmitted_bits = alice_bits.copy()
    transmitted_bases = alice_bases.copy()

    eve_idx = np.where(eve_intervenes)[0]
    if len(eve_idx) > 0:
        eve_bases[eve_idx] = rng.choice(["Z", "X"], size=len(eve_idx), p=[0.5, 0.5])
        eve_same = eve_bases[eve_idx] == alice_bases[eve_idx]
        eve_random = rng.integers(0, 2, size=len(eve_idx))
        eve_res = np.where(eve_same, alice_bits[eve_idx], eve_random)
        eve_results[eve_idx] = eve_res
        transmitted_bits[eve_idx] = eve_res
        transmitted_bases[eve_idx] = eve_bases[eve_idx]

    bob_results = np.full(n, -1, dtype=int)
    det_idx = np.where(detected)[0]
    if len(det_idx) > 0:
        same_trans_basis = bob_bases[det_idx] == transmitted_bases[det_idx]
        random_bits = rng.integers(0, 2, size=len(det_idx))
        raw_bob = np.where(same_trans_basis, transmitted_bits[det_idx], random_bits)

        dark_only = dark_detected[det_idx] & ~photon_detected[det_idx]
        dark_random = rng.integers(0, 2, size=len(det_idx))
        raw_bob = np.where(dark_only, dark_random, raw_bob)

        # Optical error / misalignment / channel noise
        flip = rng.random(len(det_idx)) < optical_error_rate
        raw_bob = np.where(flip, 1 - raw_bob, raw_bob)
        bob_results[det_idx] = raw_bob

    basis_match = detected & (alice_bases == bob_bases)
    signal_mask = intensity_label == "u"
    z_match = basis_match & (alice_bases == "Z")
    x_match = basis_match & (alice_bases == "X")

    if protocol == "T12論文値":
        key_mask = signal_mask & z_match
        check_mask = x_match
    else:
        # fair BB84 reference: same p_st and same signal probability, but basis selection is 50/50
        key_mask = signal_mask & basis_match
        check_mask = x_match

    alice_key = alice_bits[key_mask]
    bob_key = bob_results[key_mask]
    key_len = len(alice_key)
    errors = int(np.sum(alice_key != bob_key)) if key_len > 0 else 0
    qber = errors / key_len * 100 if key_len > 0 else 0.0

    x_len = int(np.sum(check_mask))
    x_errors = int(np.sum(alice_bits[check_mask] != bob_results[check_mask])) if x_len > 0 else 0
    x_qber = x_errors / x_len * 100 if x_len > 0 else 0.0

    corrected_len = key_len if (key_len > 0 and qber <= qber_threshold) else 0
    ec_leakage = int(round(f_ec * h2(qber / 100.0) * key_len)) if key_len > 0 else 0
    final_key_len = int(corrected_len * pa_compression) if corrected_len > 0 else 0
    final_key = hash_to_bits(bits_to_string(alice_key[: min(key_len, 4096)]), final_key_len) if final_key_len > 0 else "-"

    photon_0 = int(np.sum(photon_numbers == 0))
    photon_1 = int(np.sum(photon_numbers == 1))
    photon_multi = int(np.sum(photon_numbers >= 2))

    sift_eff_theory = None
    if protocol == "T12論文値":
        sift_eff_theory = (1 - p_st) * p_u * (p_z_t12 ** 2)
    else:
        sift_eff_theory = (1 - p_st) * p_u * 0.5

    summary = {
        "Protocol": protocol,
        "送信パルス数": n,
        "安定化スロット数": int(np.sum(stabilization_slot)),
        "signal u数": int(np.sum(intensity_label == "u")),
        "decoy v数": int(np.sum(intensity_label == "v")),
        "vacuum w数": int(np.sum(intensity_label == "w")),
        "検出数": int(np.sum(detected)),
        "基底一致数": int(np.sum(basis_match)),
        "Z/Z一致数": int(np.sum(z_match)),
        "X/X一致数": int(np.sum(x_match)),
        "鍵候補長": key_len,
        "鍵候補効率[%]": key_len / n * 100,
        "理論選別効率[%]": sift_eff_theory * 100,
        "誤り数": errors,
        "QBER[%]": qber,
        "X基底QBER[%]": x_qber,
        "EC leakage[bit]": ec_leakage,
        "PA圧縮比": pa_compression,
        "最終鍵長": final_key_len,
        "最終鍵効率[%]": final_key_len / n * 100,
        "推定sifted rate[Mb/s]": DEFAULT_PULSE_RATE_HZ * key_len / n / 1e6,
        "推定secure rate[Mb/s]": DEFAULT_PULSE_RATE_HZ * final_key_len / n / 1e6,
        "0光子率[%]": photon_0 / n * 100,
        "1光子率[%]": photon_1 / n * 100,
        "多光子率[%]": photon_multi / n * 100,
        "final_key": final_key,
        "can_generate_key": corrected_len > 0,
    }

    m = min(n, max_display_rows)
    detail = pd.DataFrame({
        "No.": np.arange(1, m + 1),
        "Stabilization": np.where(stabilization_slot[:m], "○", "-"),
        "Intensity": intensity_label[:m],
        "μ": mu_values[:m],
        "Photon n": photon_numbers[:m],
        "Alice Bit": alice_bits[:m],
        "Alice Basis": alice_bases[:m],
        "Detected": np.where(detected[:m], "○", "-"),
        "Eve": np.where(eve_intervenes[:m], "○", "-"),
        "Bob Basis": bob_bases[:m],
        "Bob Result": np.where(bob_results[:m] >= 0, bob_results[:m].astype(str), "-"),
        "Basis Match": np.where(basis_match[:m], "○", "×"),
        "Key Used": np.where(key_mask[:m], "○", "-"),
        "X Check": np.where(check_mask[:m], "○", "-"),
        "Key Error": np.where(key_mask[:m] & (alice_bits[:m] != bob_results[:m]), "○", "-"),
    })

    anim_len = min(n, 256)
    anim = {
        "alice_bits": alice_bits[:anim_len].tolist(),
        "bob_results": [int(x) if x >= 0 else "-" for x in bob_results[:anim_len]],
        "eve_intervened": np.where(eve_intervenes[:anim_len], "○", "-").tolist(),
    }
    return summary, detail, anim


def bit_list_to_html(bits, current_index=None, max_len=None, bits_per_row=32, bit_size=14):
    display_bits = bits if max_len is None else bits[:max_len]
    cell_width = bit_size + 28
    cell_height = bit_size + 30
    html = f"<div style='display:grid; grid-template-columns:repeat({bits_per_row}, {cell_width}px); gap:6px; align-items:center;'>"
    for i, bit in enumerate(display_bits):
        active = i == current_index
        bg = "#2563eb" if str(bit) == "1" else "#0f766e"
        if str(bit) == "-":
            bg = "#6b7280"
        border = "4px solid #facc15" if active else "1px solid #d1d5db"
        html += f"<div style='width:{cell_width}px; height:{cell_height}px; border-radius:8px; background:{bg}; color:white; border:{border}; display:flex; align-items:center; justify-content:center; font-weight:900; font-size:{bit_size}px; box-sizing:border-box;'>{bit}</div>"
    html += "</div>"
    return html


def render_bit_motion_frame(bit, index, total, phase, eve_on, eve_hit, bob_bit, alice_bits, bob_results, show_ball=True):
    left = 5 + phase * 78
    eve_display = "block" if eve_on else "none"
    eve_color = "#fee2e2" if eve_hit else "#f3f4f6"
    eve_border = "#dc2626" if eve_hit else "#9ca3af"
    eve_label = "Eve測定" if eve_hit else "Eve待機"

    if total <= 32:
        bits_per_row, bit_size, bit_area_height = 16, 22, 190
    elif total <= 64:
        bits_per_row, bit_size, bit_area_height = 24, 18, 220
    elif total <= 128:
        bits_per_row, bit_size, bit_area_height = 32, 15, 250
    else:
        bits_per_row, bit_size, bit_area_height = 32, 14, 280

    ball_html = ""
    if show_ball:
        ball_html = f"""
        <div style='position:absolute; left:{left}%; top:68px; width:54px; height:54px; border-radius:50%; background:#111827; color:#ffffff; display:flex; align-items:center; justify-content:center; font-size:30px; font-weight:900; box-shadow:0 0 18px rgba(37,99,235,0.7); z-index:8;'>{bit}</div>
        """

    html = f"""
    <div style='border:1px solid #d1d5db; border-radius:16px; padding:18px; background:#ffffff; box-sizing:border-box;'>
      <div style='font-weight:700; margin-bottom:10px; font-size:20px;'>送信ビット {index + 1} / {total}</div>
      <div style='position:relative; height:210px; background:linear-gradient(90deg,#eff6ff,#ffffff,#f0fdf4); border-radius:14px; overflow:hidden;'>
        <div style='position:absolute; left:2%; top:50px; width:150px; height:90px; border:3px solid #2563eb; background:#dbeafe; border-radius:18px; text-align:center; padding-top:18px; font-weight:900; font-size:24px; z-index:5; box-sizing:border-box;'>Alice<br><span style='font-size:36px;'>TX</span></div>
        <div style='position:absolute; left:50%; transform:translateX(-50%); top:42px; width:150px; height:105px; border:3px solid {eve_border}; background:{eve_color}; border-radius:18px; text-align:center; padding-top:18px; font-weight:900; font-size:24px; display:{eve_display}; z-index:10; box-sizing:border-box;'>Eve<br><span style='font-size:20px;'>{eve_label}</span></div>
        <div style='position:absolute; right:2%; top:50px; width:150px; height:90px; border:3px solid #16a34a; background:#dcfce7; border-radius:18px; text-align:center; padding-top:18px; font-weight:900; font-size:24px; z-index:5; box-sizing:border-box;'>Bob<br><span style='font-size:36px;'>RX</span></div>
        <div style='position:absolute; left:180px; right:180px; top:96px; height:10px; background:#94a3b8; border-radius:99px; z-index:1;'></div>
        {ball_html}
        <div style='position:absolute; left:2%; bottom:14px; font-size:18px; z-index:5;'>Alice bit = <b>{bit}</b></div>
        <div style='position:absolute; right:2%; bottom:14px; font-size:18px; z-index:5;'>Bob result = <b>{bob_bit}</b></div>
      </div>
      <div style='margin-top:24px;'>
        <div style='font-size:22px; font-weight:800; margin-bottom:8px;'>Alice送信列</div>
        <div style='height:{bit_area_height}px; overflow-y:auto; border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa; margin-bottom:22px;'>{bit_list_to_html(alice_bits, index, max_len=total, bits_per_row=bits_per_row, bit_size=bit_size)}</div>
        <div style='font-size:22px; font-weight:800; margin-bottom:8px;'>Bob受信列</div>
        <div style='height:{bit_area_height}px; overflow-y:auto; border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa;'>{bit_list_to_html(bob_results, index, max_len=total, bits_per_row=bits_per_row, bit_size=bit_size)}</div>
      </div>
    </div>
    """
    components.html(html, height=900, scrolling=False)


def animate_bit_transmission(anim, frame_delay=0.02):
    alice_bits = anim["alice_bits"]
    bob_results = anim["bob_results"]
    eve_intervened = anim["eve_intervened"]
    total = min(len(alice_bits), len(bob_results), len(eve_intervened))
    if total <= 0:
        return
    st.subheader("0. 量子ビット送信アニメーション")
    st.caption("アニメーションは先頭256パルスまで表示します。")
    placeholder = st.empty()
    for i in range(total):
        for phase in [0.0, 0.25, 0.5, 0.75, 1.0]:
            placeholder.empty()
            with placeholder.container():
                render_bit_motion_frame(alice_bits[i], i, total, phase, any(x == "○" for x in eve_intervened), eve_intervened[i] == "○", bob_results[i], alice_bits, bob_results, True)
            time.sleep(frame_delay)
    placeholder.empty()
    with placeholder.container():
        render_bit_motion_frame(alice_bits[-1], total - 1, total, 1.0, any(x == "○" for x in eve_intervened), eve_intervened[-1] == "○", bob_results[-1], alice_bits, bob_results, False)


def show_summary(summary):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("送信", f"{summary['送信パルス数']:,}")
    c2.metric("検出", f"{summary['検出数']:,}")
    c3.metric("鍵候補", f"{summary['鍵候補長']:,}")
    c4.metric("QBER", f"{summary['QBER[%]']:.2f}%")
    c5.metric("最終鍵", f"{summary['最終鍵長']:,} bit")
    c6.metric("推定SKR", f"{summary['推定secure rate[Mb/s]']:.2f} Mb/s")


if st.button("シミュレーション実行", type="primary"):
    seed = int(time.time()) % 1_000_000_000
    bb84_summary, bb84_detail, bb84_anim = simulate_protocol("BB84参照", num_pulses, seed)
    t12_summary, t12_detail, t12_anim = simulate_protocol("T12論文値", num_pulses, seed + 1)

    if protocol_mode == "BB84参照のみ":
        summaries = [bb84_summary]
        details = {"BB84参照": bb84_detail}
        anim = bb84_anim
    elif protocol_mode == "T12論文値のみ":
        summaries = [t12_summary]
        details = {"T12論文値": t12_detail}
        anim = t12_anim
    else:
        summaries = [bb84_summary, t12_summary]
        details = {"BB84参照": bb84_detail, "T12論文値": t12_detail}
        anim = t12_anim

    if show_bit_motion:
        if num_pulses > 256:
            st.caption(f"送信パルス数は {num_pulses:,} ですが、アニメーションは先頭256パルスのみ表示します。")
        animate_bit_transmission(anim)

    st.subheader("1. 全体結果")
    if protocol_mode == "比較表示":
        st.markdown("#### BB84参照")
        show_summary(bb84_summary)
        st.markdown("#### T12論文値")
        show_summary(t12_summary)
    else:
        show_summary(summaries[0])

    st.subheader("2. 効率比較")
    compare_df = pd.DataFrame([
        {
            "プロトコル": s["Protocol"],
            "理論選別効率[%]": s["理論選別効率[%]"],
            "鍵候補効率[%]": s["鍵候補効率[%]"],
            "最終鍵効率[%]": s["最終鍵効率[%]"],
            "推定sifted rate[Mb/s]": s["推定sifted rate[Mb/s]"],
            "推定secure rate[Mb/s]": s["推定secure rate[Mb/s]"],
            "QBER[%]": s["QBER[%]"],
            "鍵候補長": s["鍵候補長"],
            "最終鍵長": s["最終鍵長"],
        }
        for s in [bb84_summary, t12_summary]
    ])

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.bar(compare_df, x="プロトコル", y=["理論選別効率[%]", "鍵候補効率[%]", "最終鍵効率[%]"], barmode="group", text_auto=".2f", title="効率比較")
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = px.bar(compare_df, x="プロトコル", y=["推定sifted rate[Mb/s]", "推定secure rate[Mb/s]"], barmode="group", text_auto=".2f", title="1GHz換算レート")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(compare_df, use_container_width=True)

    gain = t12_summary["理論選別効率[%]"] / bb84_summary["理論選別効率[%]"] if bb84_summary["理論選別効率[%]"] > 0 else 0
    st.info(f"論文値ベースの理論選別効率では、T12はBB84参照の約 {gain:.2f} 倍です。T12の eta_sift は約 {t12_summary['理論選別効率[%]']:.2f}% です。")

    st.subheader("3. 強度選択・光子数分布")
    photon_df = pd.DataFrame([
        {"プロトコル": s["Protocol"], "分類": "0光子", "割合[%]": s["0光子率[%]"]} for s in [bb84_summary, t12_summary]
    ] + [
        {"プロトコル": s["Protocol"], "分類": "1光子", "割合[%]": s["1光子率[%]"]} for s in [bb84_summary, t12_summary]
    ] + [
        {"プロトコル": s["Protocol"], "分類": "多光子", "割合[%]": s["多光子率[%]"]} for s in [bb84_summary, t12_summary]
    ])
    fig3 = px.bar(photon_df, x="分類", y="割合[%]", color="プロトコル", barmode="group", text_auto=".2f", title="全パルスの光子数分布")
    st.plotly_chart(fig3, use_container_width=True)

    intensity_df = pd.DataFrame([
        {"プロトコル": s["Protocol"], "分類": "signal u", "数": s["signal u数"]} for s in [bb84_summary, t12_summary]
    ] + [
        {"プロトコル": s["Protocol"], "分類": "decoy v", "数": s["decoy v数"]} for s in [bb84_summary, t12_summary]
    ] + [
        {"プロトコル": s["Protocol"], "分類": "vacuum w", "数": s["vacuum w数"]} for s in [bb84_summary, t12_summary]
    ] + [
        {"プロトコル": s["Protocol"], "分類": "stabilization", "数": s["安定化スロット数"]} for s in [bb84_summary, t12_summary]
    ])
    fig4 = px.bar(intensity_df, x="分類", y="数", color="プロトコル", barmode="group", text_auto=True, title="強度選択・安定化スロット数")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("4. 計算式")
    with st.expander("T12論文値の式", expanded=True):
        st.markdown(f"""
        ### 選別効率

        ```text
        eta_sift = (1 - p_st) * p_u * p_Z^2
        ```

        今回の値：

        ```text
        eta_sift = (1 - {p_st:.6f}) * {p_u:.5f} * {p_z_t12:.5f}^2
                 = {((1-p_st)*p_u*(p_z_t12**2)):.5f}
                 = {((1-p_st)*p_u*(p_z_t12**2))*100:.2f} %
        ```

        ### 光子数分布

        各パルスで強度 u/v/w を選んだ後、その平均光子数 μ に対して以下で光子数を生成します。

        ```text
        n ~ Poisson(μ)
        P(n, μ) = exp(-μ) * μ^n / n!
        ```

        ### 検出確率

        ```text
        eta_i = 1 - (1 - eta)^i
        ```

        ### 暗計数を含む簡略検出

        ```text
        detected = photon_detected OR dark_detected
        ```

        ### 誤り訂正・秘匿性増幅

        ```text
        EC leakage ≒ fEC * H2(QBER) * key_length
        final_key_length ≒ corrected_key_length * PA compression ratio
        ```

        今回は代表値として、

        ```text
        fEC = {f_ec:.2f}
        PA compression ratio = {pa_compression:.2f}
        ```

        を使っています。
        """)

    st.subheader("5. 最終鍵")
    for s in summaries:
        with st.expander(f"{s['Protocol']} の最終鍵", expanded=False):
            if s["can_generate_key"] and s["最終鍵長"] > 0:
                st.success("鍵生成成功")
                st.code(s["final_key"], language="text")
            else:
                st.error("QBERが高い、または鍵候補が不足しているため、最終鍵は生成されませんでした。")

    if show_detail_table:
        st.subheader("6. 詳細表")
        for name, df in details.items():
            with st.expander(f"{name} 詳細表", expanded=(protocol_mode != "比較表示")):
                st.caption(f"先頭 {len(df):,} 行のみ表示しています。計算は全 {num_pulses:,} パルスで実行しています。")
                st.dataframe(df, use_container_width=True)
else:
    st.info("左の条件を設定して、［シミュレーション実行］を押してください。")
