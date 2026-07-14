import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import time
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(page_title="BB84 / T12 full-ish QKD Simulator", layout="wide")

st.title("BB84 / T12 論文値 + デコイ・有限サイズ・EC/PA 簡略モデル")
st.caption("T12論文値を反映し、デコイ解析・有限サイズ補正・ECリーク・認証コスト・位相誤り推定・復号失敗率を教育用に近似")

# Paper/default values
DEFAULT_PULSE_RATE_HZ = 1_000_000_000
DEFAULT_MU_U = 0.4
DEFAULT_MU_V = 0.1
DEFAULT_MU_W = 0.0007
DEFAULT_P_U = 0.96973
DEFAULT_P_V = 0.01661
DEFAULT_P_W = 0.01466
DEFAULT_P_Z = 0.96677
DEFAULT_P_ST = 1 / 128
DEFAULT_EPSILON_SEC = 1e-10
DEFAULT_EPSILON_AUTH = 1e-10
DEFAULT_F_EC_AT_3PCT = 1.34
DEFAULT_EC_FAIL_PROB = 0.0073
DEFAULT_PA_DATASET_BITS = 100.66e6

st.markdown("""
このアプリは、T12論文値を使いながら、以下の後処理要素を簡略的に入れたモデルです。

- デコイ解析：signal/decoy/vacuum の検出率から単一光子寄与を推定
- 有限サイズ補正：X基底サンプル数に応じて位相誤りにマージンを追加
- ECリーク：LDPC風に `fEC * n * H2(QBER)` で公開情報量を見積もり
- 認証コスト：`2 log2(1/epsilon_auth)` bit を差し引き
- Phase error estimation：X基底情報からZ基底鍵の位相誤り率を推定
- 復号失敗率：EC失敗確率に応じて平均的な通過ブロック割合を反映

※ 実際のLDPC復号や厳密な有限鍵セキュリティ証明そのものではなく、論文値の傾向を確認するための教育用近似です。
""")

with st.expander("T12論文値・後処理モデルの前提", expanded=True):
    st.markdown(f"""
    ```text
    f = 1 GHz
    u,v,w = 0.4, 0.1, 0.0007 photons/pulse
    p_u,p_v,p_w = 96.973%, 1.661%, 1.466%
    p_Z = 96.677%, p_X = 3.323%
    p_st = 1/128
    PA dataset = 100.66 Mb
    epsilon = 1e-10
    EC failure probability ≈ 0.73%
    代表的 fEC ≈ 1.32 - 1.36 around QBER 3%
    ```
    """)

with st.sidebar:
    st.header("シミュレーション条件")
    num_pulses = st.select_slider(
        "送信パルス数",
        options=[32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576],
        value=1048576,
    )
    protocol_mode = st.radio("表示モード", ["比較表示", "BB84参照のみ", "T12論文値のみ"], index=0)

    st.markdown("---")
    st.subheader("T12論文値")
    p_z_t12 = DEFAULT_P_Z
    p_x_t12 = 1 - p_z_t12
    mu_u = DEFAULT_MU_U
    mu_v = DEFAULT_MU_V
    mu_w = DEFAULT_MU_W
    p_u = DEFAULT_P_U
    p_v = DEFAULT_P_V
    p_w = DEFAULT_P_W
    p_st = DEFAULT_P_ST

    st.caption(f"pZ={p_z_t12*100:.3f}%, pX={p_x_t12*100:.3f}%, u/v/w={mu_u}/{mu_v}/{mu_w}")

    st.markdown("---")
    st.subheader("物理・検出条件")
    total_detection_efficiency = st.slider("総合検出効率 η [%]", 1.0, 100.0, 31.0, step=1.0) / 100.0
    dark_count_rate = st.slider("暗計数率 Y0 [%/pulse]", 0.0, 5.0, 0.045, step=0.001) / 100.0
    optical_error_rate = st.slider("光学系・通信路由来の誤り率 Eopt [%]", 0.0, 20.0, 3.0, step=0.1) / 100.0
    afterpulse_error_rate = st.slider("アフターパルス由来の追加誤り率 [%]", 0.0, 10.0, 0.0, step=0.1) / 100.0

    eve_enabled = st.checkbox("Eveによる遮断・再送信攻撃を有効化", value=False)
    eve_rate = st.slider("Eve介入率 [%]", 0, 100, 0, step=5, disabled=not eve_enabled) / 100.0

    st.markdown("---")
    st.subheader("後処理条件")
    qber_threshold = st.slider("鍵破棄しきい値 QBER [%]", 0.0, 20.0, 11.0, step=0.5)
    finite_sigma = st.slider("有限サイズ補正の強さ sigma", 0.0, 10.0, 5.0, step=0.5)
    epsilon_auth = DEFAULT_EPSILON_AUTH
    ec_fail_prob = st.slider("EC復号失敗率 [%]", 0.0, 5.0, DEFAULT_EC_FAIL_PROB * 100, step=0.01) / 100.0
    ec_model = st.radio("ECモデル", ["LDPC風・QBER依存", "固定fEC"], index=0)
    fixed_f_ec = st.slider("固定 fEC", 1.00, 2.00, DEFAULT_F_EC_AT_3PCT, step=0.01)

    show_bit_motion = st.checkbox("0/1送信アニメーションを表示", value=False)
    show_detail_table = st.checkbox("詳細表を表示", value=True)
    max_display_rows = st.slider("詳細表の最大行数", 50, 1000, 500, step=50)


def h2(q):
    q = min(max(float(q), 1e-12), 1 - 1e-12)
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)


def f_ec_ldpc_like(qber):
    q = qber * 100
    # Paper-like rough interpolation: around 3% -> 1.34, higher QBER -> more leakage
    if q <= 2:
        return 1.20
    if q <= 3:
        return 1.20 + (q - 2) * (1.34 - 1.20) / 1
    if q <= 5:
        return 1.34 + (q - 3) * (1.38 - 1.34) / 2
    if q <= 10:
        return 1.38 + (q - 5) * (1.50 - 1.38) / 5
    return 1.60


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
    pz = 0.5 if protocol == "BB84参照" else p_z_t12
    alice = rng.choice(["Z", "X"], size=n, p=[pz, 1 - pz])
    bob = rng.choice(["Z", "X"], size=n, p=[pz, 1 - pz])
    return alice, bob


def choose_intensities(rng, n):
    probs = np.array([p_u, p_v, p_w], dtype=float)
    probs = probs / probs.sum()
    labels = rng.choice(["u", "v", "w"], size=n, p=probs)
    mu = np.where(labels == "u", mu_u, np.where(labels == "v", mu_v, mu_w))
    return labels, mu


def decoy_estimate(counts):
    # Vacuum + weak decoy lower bound, educational approximation
    mu = mu_u
    nu = mu_v
    omega = mu_w

    N_u = max(counts["N_u"], 1)
    N_v = max(counts["N_v"], 1)
    N_w = max(counts["N_w"], 1)
    Q_u = counts["D_u"] / N_u
    Q_v = counts["D_v"] / N_v
    Q_w = counts["D_w"] / N_w
    E_u = counts["Err_u"] / max(counts["D_u"], 1)
    E_v = counts["Err_v"] / max(counts["D_v"], 1)
    E_w = counts["Err_w"] / max(counts["D_w"], 1)

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

    return {
        "Q_u": Q_u,
        "Q_v": Q_v,
        "Q_w": Q_w,
        "E_u": E_u,
        "E_v": E_v,
        "E_w": E_w,
        "Y0_est": Y0,
        "Y1_L": Y1_L,
        "Q1_L": Q1_L,
        "e1_U": e1_U,
    }


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

    eve_intervenes = eve_enabled & (rng.random(n) < eve_rate) & detected
    eve_bases = np.full(n, "-", dtype=object)
    transmitted_bits = alice_bits.copy()
    transmitted_bases = alice_bases.copy()

    eve_idx = np.where(eve_intervenes)[0]
    if len(eve_idx) > 0:
        eve_bases[eve_idx] = rng.choice(["Z", "X"], size=len(eve_idx), p=[0.5, 0.5])
        eve_same = eve_bases[eve_idx] == alice_bases[eve_idx]
        eve_random = rng.integers(0, 2, size=len(eve_idx))
        eve_res = np.where(eve_same, alice_bits[eve_idx], eve_random)
        transmitted_bits[eve_idx] = eve_res
        transmitted_bases[eve_idx] = eve_bases[eve_idx]

    bob_results = np.full(n, -1, dtype=int)
    det_idx = np.where(detected)[0]
    if len(det_idx) > 0:
        same_trans_basis = bob_bases[det_idx] == transmitted_bases[det_idx]
        random_bits = rng.integers(0, 2, size=len(det_idx))
        raw_bob = np.where(same_trans_basis, transmitted_bits[det_idx], random_bits)
        dark_only = dark_detected[det_idx] & ~photon_detected[det_idx]
        raw_bob = np.where(dark_only, rng.integers(0, 2, size=len(det_idx)), raw_bob)
        flip_prob = min(1.0, optical_error_rate + afterpulse_error_rate)
        flip = rng.random(len(det_idx)) < flip_prob
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
        key_mask = signal_mask & basis_match
        check_mask = x_match

    alice_key = alice_bits[key_mask]
    bob_key = bob_results[key_mask]
    key_len = len(alice_key)
    key_errors = int(np.sum(alice_key != bob_key)) if key_len > 0 else 0
    qber = key_errors / key_len if key_len > 0 else 0.0

    x_len = int(np.sum(check_mask))
    x_errors = int(np.sum(alice_bits[check_mask] != bob_results[check_mask])) if x_len > 0 else 0
    x_qber = x_errors / x_len if x_len > 0 else 0.0

    # Decoy counts in matched/key-relevant subset, per intensity
    def counts_for(label):
        m = basis_match & (intensity_label == label)
        D = int(np.sum(m))
        Err = int(np.sum((alice_bits[m] != bob_results[m]))) if D > 0 else 0
        N = int(np.sum((intensity_label == label) & (~stabilization_slot)))
        return N, D, Err

    N_u, D_u, Err_u = counts_for("u")
    N_v, D_v, Err_v = counts_for("v")
    N_w, D_w, Err_w = counts_for("w")
    decoy = decoy_estimate({"N_u": N_u, "D_u": D_u, "Err_u": Err_u, "N_v": N_v, "D_v": D_v, "Err_v": Err_v, "N_w": N_w, "D_w": D_w, "Err_w": Err_w})

    # Phase error estimation: X basis + decoy single-photon upper error + finite margin
    finite_margin = finite_sigma / np.sqrt(max(x_len, 1))
    phase_error_est = min(0.5, max(x_qber, decoy["e1_U"]) + finite_margin)

    f_ec = f_ec_ldpc_like(qber) if ec_model == "LDPC風・QBER依存" else fixed_f_ec
    ec_leakage = int(round(f_ec * h2(qber) * key_len)) if key_len > 0 else 0
    auth_cost = int(np.ceil(2 * np.log2(1 / epsilon_auth)))

    # Finite-size security penalty; coarse educational approximation
    finite_penalty = int(np.ceil(finite_sigma * np.sqrt(max(key_len, 1)) * np.log2(max(key_len, 2)))) if key_len > 0 else 0

    # Single-photon lower-bound key contribution scaled by observed key-mask opportunity.
    # For educational stability, combine actual sifted key length with phase error estimate.
    privacy_term = int(round(key_len * h2(phase_error_est))) if key_len > 0 else 0
    raw_secure = key_len - ec_leakage - privacy_term - finite_penalty - auth_cost
    after_failure = int(max(0, raw_secure) * (1 - ec_fail_prob))

    can_generate_key = key_len > 0 and (qber * 100) <= qber_threshold and after_failure > 0
    final_key_len = after_failure if can_generate_key else 0
    final_key = hash_to_bits(bits_to_string(alice_key[: min(key_len, 4096)]), final_key_len) if final_key_len > 0 else "-"

    if protocol == "T12論文値":
        theory_sift = (1 - p_st) * p_u * p_z_t12 ** 2
    else:
        theory_sift = (1 - p_st) * p_u * 0.5

    # Theoretical paper-like estimate based on 100.66 Mb - shown for reference
    photon_0 = int(np.sum(photon_numbers == 0))
    photon_1 = int(np.sum(photon_numbers == 1))
    photon_multi = int(np.sum(photon_numbers >= 2))

    summary = {
        "Protocol": protocol,
        "送信パルス数": n,
        "検出数": int(np.sum(detected)),
        "基底一致数": int(np.sum(basis_match)),
        "Z/Z一致数": int(np.sum(z_match)),
        "X/X一致数": int(np.sum(x_match)),
        "鍵候補長": key_len,
        "鍵候補効率[%]": key_len / n * 100,
        "理論選別効率[%]": theory_sift * 100,
        "誤り数": key_errors,
        "QBER[%]": qber * 100,
        "X基底QBER[%]": x_qber * 100,
        "phase error[%]": phase_error_est * 100,
        "fEC": f_ec,
        "EC leakage[bit]": ec_leakage,
        "privacy term[bit]": privacy_term,
        "finite penalty[bit]": finite_penalty,
        "auth cost[bit]": auth_cost,
        "EC fail prob[%]": ec_fail_prob * 100,
        "最終鍵長": final_key_len,
        "最終鍵効率[%]": final_key_len / n * 100,
        "推定sifted rate[Mb/s]": DEFAULT_PULSE_RATE_HZ * key_len / n / 1e6,
        "推定secure rate[Mb/s]": DEFAULT_PULSE_RATE_HZ * final_key_len / n / 1e6,
        "0光子率[%]": photon_0 / n * 100,
        "1光子率[%]": photon_1 / n * 100,
        "多光子率[%]": photon_multi / n * 100,
        "Y0_est": decoy["Y0_est"],
        "Y1_L": decoy["Y1_L"],
        "Q1_L": decoy["Q1_L"],
        "e1_U[%]": decoy["e1_U"] * 100,
        "final_key": final_key,
        "can_generate_key": can_generate_key,
    }

    m = min(n, max_display_rows)
    detail = pd.DataFrame({
        "No.": np.arange(1, m + 1),
        "Intensity": intensity_label[:m],
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
    bits_per_row, bit_size, bit_area_height = 32, 14, 280
    ball_html = f"<div style='position:absolute; left:{left}%; top:68px; width:54px; height:54px; border-radius:50%; background:#111827; color:#ffffff; display:flex; align-items:center; justify-content:center; font-size:30px; font-weight:900; box-shadow:0 0 18px rgba(37,99,235,0.7); z-index:8;'>{bit}</div>" if show_ball else ""
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


def show_summary(s):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("送信", f"{s['送信パルス数']:,}")
    c2.metric("検出", f"{s['検出数']:,}")
    c3.metric("鍵候補", f"{s['鍵候補長']:,}")
    c4.metric("QBER", f"{s['QBER[%]']:.2f}%")
    c5.metric("最終鍵", f"{s['最終鍵長']:,} bit")
    c6.metric("推定SKR", f"{s['推定secure rate[Mb/s]']:.2f} Mb/s")


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
        for s in [bb84_summary, t12_summary]
    ])
    st.dataframe(post_df, use_container_width=True)

    fig_post = px.bar(post_df, x="プロトコル", y=["EC leakage[bit]", "privacy term[bit]", "finite penalty[bit]", "最終鍵長"], barmode="group", text_auto=True, title="後処理による鍵長の内訳")
    st.plotly_chart(fig_post, use_container_width=True)

    st.subheader("3. デコイ解析・Phase error estimation")
    decoy_df = pd.DataFrame([
        {
            "プロトコル": s["Protocol"],
            "Y0_est": s["Y0_est"],
            "Y1_L": s["Y1_L"],
            "Q1_L": s["Q1_L"],
            "e1_U[%]": s["e1_U[%]"],
            "X基底QBER[%]": s["X基底QBER[%]"],
            "phase error[%]": s["phase error[%]"],
        }
        for s in [bb84_summary, t12_summary]
    ])
    st.dataframe(decoy_df, use_container_width=True)

    st.subheader("4. 効率比較")
    compare_df = pd.DataFrame([
        {
            "プロトコル": s["Protocol"],
            "理論選別効率[%]": s["理論選別効率[%]"],
            "鍵候補効率[%]": s["鍵候補効率[%]"],
            "最終鍵効率[%]": s["最終鍵効率[%]"],
            "推定sifted rate[Mb/s]": s["推定sifted rate[Mb/s]"],
            "推定secure rate[Mb/s]": s["推定secure rate[Mb/s]"],
        }
        for s in [bb84_summary, t12_summary]
    ])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(compare_df, x="プロトコル", y=["理論選別効率[%]", "鍵候補効率[%]", "最終鍵効率[%]"], barmode="group", text_auto=".2f"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(compare_df, x="プロトコル", y=["推定sifted rate[Mb/s]", "推定secure rate[Mb/s]"], barmode="group", text_auto=".2f"), use_container_width=True)
    st.dataframe(compare_df, use_container_width=True)

    st.subheader("5. 光子数分布")
    photon_df = pd.DataFrame([
        {"プロトコル": s["Protocol"], "分類": "0光子", "割合[%]": s["0光子率[%]"]} for s in [bb84_summary, t12_summary]
    ] + [
        {"プロトコル": s["Protocol"], "分類": "1光子", "割合[%]": s["1光子率[%]"]} for s in [bb84_summary, t12_summary]
    ] + [
        {"プロトコル": s["Protocol"], "分類": "多光子", "割合[%]": s["多光子率[%]"]} for s in [bb84_summary, t12_summary]
    ])
    st.plotly_chart(px.bar(photon_df, x="分類", y="割合[%]", color="プロトコル", barmode="group", text_auto=".2f"), use_container_width=True)

    st.subheader("6. 計算式")
    with st.expander("追加したモデルの式", expanded=True):
        st.markdown(f"""
        ### デコイ解析

        ```text
        Q_mu, Q_nu, Q_w : signal / decoy / vacuum の検出率
        Y0 ≈ Q_w
        Y1_L : 単一光子 yield の下限
        e1_U : 単一光子誤り率の上限
        ```

        ### Phase error estimation

        ```text
        phase_error = max(X基底QBER, e1_U) + sigma / sqrt(N_X)
        ```

        今回の sigma は `{finite_sigma:.1f}` です。

        ### ECリーク

        ```text
        leak_EC = fEC * H2(QBER) * 鍵候補長
        ```

        ECモデルは `{ec_model}` です。

        ### 認証コスト

        ```text
        auth_cost = 2 * log2(1 / epsilon_auth)
        ```

        ### 最終鍵長

        ```text
        final_key = key_length
                    - EC leakage
                    - privacy term
                    - finite penalty
                    - auth cost
        final_key *= (1 - EC failure probability)
        ```
        """)

    st.subheader("7. 最終鍵")
    for s in summaries:
        with st.expander(f"{s['Protocol']} の最終鍵", expanded=False):
            if s["can_generate_key"] and s["最終鍵長"] > 0:
                st.success("鍵生成成功")
                st.code(s["final_key"], language="text")
            else:
                st.error("QBERが高い、または後処理後の鍵長が0以下のため、最終鍵は生成されませんでした。")

    if show_detail_table:
        st.subheader("8. 詳細表")
        for name, df in details.items():
            with st.expander(f"{name} 詳細表", expanded=(protocol_mode != "比較表示")):
                st.caption(f"先頭 {len(df):,} 行のみ表示しています。計算は全 {num_pulses:,} パルスで実行しています。")
                st.dataframe(df, use_container_width=True)
else:
    st.info("左の条件を設定して、［シミュレーション実行］を押してください。")
