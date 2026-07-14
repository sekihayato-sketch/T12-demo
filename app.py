import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import time
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(page_title="BB84 / T12-like QKD Simulator", layout="wide")

st.title("BB84 / T12風 量子鍵配送シミュレータ")
st.caption("従来BB84と、Z基底を高確率で使うT12風モデルを比較する教育用アプリ")

st.markdown("""
このアプリでは、従来のBB84モデルと、**Z基底を 15/16、X基底を 1/16** の確率で選ぶ
**T12風の高効率モデル**を比較できます。

また、各パルスの光子数は **平均光子数 μ のポアソン分布**に従って生成します。
デフォルトでは μ = 1.0 です。

※ ここでの「T12風」は、指定いただいた **基底選択比 15/16 : 1/16** と **ポアソン光子数分布**を反映した教育用モデルです。実際の社内実装・製品実装そのものを表すものではありません。
""")

with st.expander("このモデルで表していること", expanded=False):
    st.markdown("""
    ### 従来BB84モデル
    - Aliceの基底：Z/Xを50:50で選択
    - Bobの基底：Z/Xを50:50で選択
    - AliceとBobの基底が一致した検出イベントを鍵候補にする

    ### T12風モデル
    - Aliceの基底：Zを15/16、Xを1/16で選択
    - Bobの基底：Zを15/16、Xを1/16で選択
    - Z/Z一致の検出イベントを主な鍵候補にする
    - X/X一致は主に誤り率確認・パラメータ推定用として扱う

    ### 光子数モデル
    各パルスの光子数 n は以下で生成します。

    ```text
    n ~ Poisson(μ)
    ```

    μ = 1.0 の場合、0光子、1光子、2光子以上のパルスが確率的に混ざります。
    このアプリでは、検出確率を簡略的に次のように置いています。

    ```text
    検出確率 = 1 - (1 - η)^n
    ```

    ここで η はチャネル・受信器を含む簡略的な総合検出効率です。
    """)

with st.sidebar:
    st.header("シミュレーション条件")

    num_bits = st.select_slider(
        "送信パルス数",
        options=[
            32,
            64,
            128,
            256,
            512,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
        ],
        value=4096,
    )

    protocol_mode = st.radio(
        "表示モード",
        ["比較表示", "従来BB84のみ", "T12風のみ"],
        index=0,
    )

    mean_photon_mu = st.slider("平均光子数 μ", 0.01, 2.00, 1.00, step=0.01)
    detection_efficiency = st.slider("総合検出効率 η [%]", 1.0, 100.0, 20.0, step=1.0) / 100.0
    dark_count_rate = st.slider("暗計数率 [%/pulse]", 0.0, 5.0, 0.0, step=0.1) / 100.0
    noise_rate = st.slider("通信路ノイズ率 [%]", 0.0, 20.0, 1.0, step=0.5) / 100.0

    eve_enabled = st.checkbox("Eveによる盗聴を有効化", value=False)
    eve_rate = st.slider("Eveが介入する割合 [%]", 0, 100, 0, step=5, disabled=not eve_enabled) / 100.0

    qber_threshold = st.slider("鍵破棄しきい値 QBER [%]", 0.0, 20.0, 11.0, step=0.5)

    show_bit_motion = st.checkbox("0/1ビットの送信アニメーションを表示", value=True)
    show_detail_table = st.checkbox("詳細表を表示", value=True)
    max_display_rows = st.slider("詳細表に表示する最大行数", 50, 1000, 500, step=50)


# -----------------------------
# Utility functions
# -----------------------------

def bits_to_string(bits):
    if len(bits) == 0:
        return "-"
    text = "".join(str(int(b)) for b in bits)
    return text


def short_bits(bits, max_len=96):
    text = bits_to_string(bits)
    if len(text) > max_len:
        return text[:max_len] + " ..."
    return text


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
    if protocol == "BB84":
        alice = rng.choice(["Z", "X"], size=n, p=[0.5, 0.5])
        bob = rng.choice(["Z", "X"], size=n, p=[0.5, 0.5])
    else:
        alice = rng.choice(["Z", "X"], size=n, p=[15 / 16, 1 / 16])
        bob = rng.choice(["Z", "X"], size=n, p=[15 / 16, 1 / 16])
    return alice, bob


def simulate_protocol(
    protocol,
    num_pulses,
    mu,
    eta,
    dark_rate,
    channel_noise,
    eve_on,
    eve_probability,
    qber_limit,
    seed,
):
    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(0, 2, size=num_pulses)
    alice_bases, bob_bases = choose_bases(rng, num_pulses, protocol)

    photon_numbers = rng.poisson(mu, size=num_pulses)
    photon_detection_prob = 1.0 - np.power(1.0 - eta, photon_numbers)
    photon_detected = rng.random(num_pulses) < photon_detection_prob
    dark_detected = rng.random(num_pulses) < dark_rate
    detected = photon_detected | dark_detected

    eve_intervenes = eve_on & (rng.random(num_pulses) < eve_probability)
    eve_bases = np.full(num_pulses, "-", dtype=object)
    eve_results = np.full(num_pulses, -1, dtype=int)

    transmitted_bits = alice_bits.copy()
    transmitted_bases = alice_bases.copy()

    eve_indices = np.where(eve_intervenes)[0]
    if len(eve_indices) > 0:
        eve_bases[eve_indices] = rng.choice(["Z", "X"], size=len(eve_indices), p=[0.5, 0.5])
        eve_same = eve_bases[eve_indices] == alice_bases[eve_indices]
        random_eve_bits = rng.integers(0, 2, size=len(eve_indices))
        eve_results_part = np.where(eve_same, alice_bits[eve_indices], random_eve_bits)
        eve_results[eve_indices] = eve_results_part
        transmitted_bits[eve_indices] = eve_results_part
        transmitted_bases[eve_indices] = eve_bases[eve_indices]

    bob_results = np.full(num_pulses, -1, dtype=int)

    detected_indices = np.where(detected)[0]
    if len(detected_indices) > 0:
        same_transmitted_basis = bob_bases[detected_indices] == transmitted_bases[detected_indices]
        random_bob_bits = rng.integers(0, 2, size=len(detected_indices))
        bob_detected_results = np.where(
            same_transmitted_basis,
            transmitted_bits[detected_indices],
            random_bob_bits,
        )

        # Dark count only event: result is random
        dark_only = dark_detected[detected_indices] & ~photon_detected[detected_indices]
        dark_random_bits = rng.integers(0, 2, size=len(detected_indices))
        bob_detected_results = np.where(dark_only, dark_random_bits, bob_detected_results)

        # Channel noise / receiver error: flip Bob result
        noise_flip = rng.random(len(detected_indices)) < channel_noise
        bob_detected_results = np.where(noise_flip, 1 - bob_detected_results, bob_detected_results)
        bob_results[detected_indices] = bob_detected_results

    basis_match = detected & (alice_bases == bob_bases)
    z_match = basis_match & (alice_bases == "Z")
    x_match = basis_match & (alice_bases == "X")

    if protocol == "T12風":
        key_mask = z_match
        check_mask = x_match
    else:
        key_mask = basis_match
        check_mask = x_match

    alice_key = alice_bits[key_mask]
    bob_key = bob_results[key_mask]

    key_length = len(alice_key)
    key_errors = int(np.sum(alice_key != bob_key)) if key_length > 0 else 0
    qber = key_errors / key_length * 100 if key_length > 0 else 0.0

    x_check_len = int(np.sum(check_mask))
    x_errors = int(np.sum(alice_bits[check_mask] != bob_results[check_mask])) if x_check_len > 0 else 0
    x_qber = x_errors / x_check_len * 100 if x_check_len > 0 else 0.0

    photon_0 = int(np.sum(photon_numbers == 0))
    photon_1 = int(np.sum(photon_numbers == 1))
    photon_multi = int(np.sum(photon_numbers >= 2))

    can_generate_key = key_length > 0 and qber <= qber_limit
    ec_leakage_bits = key_errors if can_generate_key else 0
    safety_factor = max(0.0, 1.0 - qber / qber_limit) if qber_limit > 0 else 0.0

    if can_generate_key:
        final_key_length = max(0, int((key_length - ec_leakage_bits) * safety_factor))
        final_key = hash_to_bits(bits_to_string(alice_key[: min(len(alice_key), 2048)]), final_key_length)
    else:
        final_key_length = 0
        final_key = "-"

    summary = {
        "Protocol": protocol,
        "送信パルス数": num_pulses,
        "検出数": int(np.sum(detected)),
        "検出率[%]": float(np.sum(detected) / num_pulses * 100),
        "基底一致数": int(np.sum(basis_match)),
        "Z/Z一致数": int(np.sum(z_match)),
        "X/X一致数": int(np.sum(x_match)),
        "鍵候補長": key_length,
        "鍵候補効率[%]": float(key_length / num_pulses * 100),
        "誤り数": key_errors,
        "QBER[%]": qber,
        "X基底QBER[%]": x_qber,
        "EC leakage[bit]": ec_leakage_bits,
        "safety_factor": safety_factor,
        "最終鍵長": final_key_length,
        "最終鍵効率[%]": float(final_key_length / num_pulses * 100),
        "0光子数": photon_0,
        "1光子数": photon_1,
        "多光子数": photon_multi,
        "0光子率[%]": photon_0 / num_pulses * 100,
        "1光子率[%]": photon_1 / num_pulses * 100,
        "多光子率[%]": photon_multi / num_pulses * 100,
        "final_key": final_key,
        "can_generate_key": can_generate_key,
    }

    detail_limit = min(num_pulses, max_display_rows)
    detail_df = pd.DataFrame({
        "No.": np.arange(1, detail_limit + 1),
        "Alice Bit": alice_bits[:detail_limit],
        "Alice Basis": alice_bases[:detail_limit],
        "Photon n": photon_numbers[:detail_limit],
        "Detected": np.where(detected[:detail_limit], "○", "-"),
        "Eve介入": np.where(eve_intervenes[:detail_limit], "○", "-"),
        "Eve Basis": eve_bases[:detail_limit],
        "Eve Result": np.where(eve_results[:detail_limit] >= 0, eve_results[:detail_limit].astype(str), "-"),
        "Bob Basis": bob_bases[:detail_limit],
        "Bob Result": np.where(bob_results[:detail_limit] >= 0, bob_results[:detail_limit].astype(str), "-"),
        "Basis Match": np.where(basis_match[:detail_limit], "○", "×"),
        "Z Key": np.where(key_mask[:detail_limit], "○", "-"),
        "X Check": np.where(check_mask[:detail_limit], "○", "-"),
        "Key Error": np.where(key_mask[:detail_limit] & (alice_bits[:detail_limit] != bob_results[:detail_limit]), "○", "-"),
    })

    animation_payload = {
        "alice_bits": alice_bits[: min(num_pulses, 256)].tolist(),
        "bob_results": [int(x) if x >= 0 else "-" for x in bob_results[: min(num_pulses, 256)]],
        "eve_intervened": np.where(eve_intervenes[: min(num_pulses, 256)], "○", "-").tolist(),
    }

    return summary, detail_df, animation_payload


def bit_list_to_html(bits, current_index=None, max_len=None, bits_per_row=32, bit_size=14):
    if max_len is None:
        display_bits = bits
    else:
        display_bits = bits[:max_len]

    cell_width = bit_size + 28
    cell_height = bit_size + 30

    html = (
        f"<div style='display:grid; grid-template-columns:repeat({bits_per_row}, {cell_width}px); "
        f"gap:6px; align-items:center;'>"
    )

    for i, bit in enumerate(display_bits):
        active = i == current_index
        bg = "#2563eb" if str(bit) == "1" else "#0f766e"
        if str(bit) == "-":
            bg = "#6b7280"
        border = "4px solid #facc15" if active else "1px solid #d1d5db"
        html += (
            f"<div style='width:{cell_width}px; height:{cell_height}px; border-radius:8px; "
            f"background:{bg}; color:white; border:{border}; display:flex; align-items:center; "
            f"justify-content:center; font-weight:900; font-size:{bit_size}px; box-sizing:border-box;'>"
            f"{bit}</div>"
        )

    html += "</div>"
    return html


def render_bit_motion_frame(bit, index, total, phase, eve_on, eve_hit, bob_bit, alice_bits, bob_results, show_ball=True):
    left = 5 + phase * 78
    eve_display = "block" if eve_on else "none"
    eve_color = "#fee2e2" if eve_hit else "#f3f4f6"
    eve_border = "#dc2626" if eve_hit else "#9ca3af"
    eve_label = "Eve測定" if eve_hit else "Eve待機"

    if total <= 32:
        bits_per_row = 16
        bit_size = 22
        bit_area_height = 190
    elif total <= 64:
        bits_per_row = 24
        bit_size = 18
        bit_area_height = 220
    elif total <= 128:
        bits_per_row = 32
        bit_size = 15
        bit_area_height = 250
    else:
        bits_per_row = 32
        bit_size = 14
        bit_area_height = 280

    ball_html = ""
    if show_ball:
        ball_html = f"""
        <div style='position:absolute; left:{left}%; top:68px; width:54px; height:54px;
                    border-radius:50%; background:#111827; color:#ffffff;
                    display:flex; align-items:center; justify-content:center;
                    font-size:30px; font-weight:900;
                    box-shadow:0 0 18px rgba(37,99,235,0.7); z-index:8;'>
          {bit}
        </div>
        """

    html = f"""
    <div style='border:1px solid #d1d5db; border-radius:16px; padding:18px; background:#ffffff; box-sizing:border-box;'>
      <div style='font-weight:700; margin-bottom:10px; font-size:20px;'>
        送信ビット {index + 1} / {total}：AliceからBobへ量子状態を送信中
      </div>

      <div style='position:relative; height:210px; background:linear-gradient(90deg,#eff6ff,#ffffff,#f0fdf4); border-radius:14px; overflow:hidden;'>
        <div style='position:absolute; left:2%; top:50px; width:150px; height:90px; border:3px solid #2563eb; background:#dbeafe; border-radius:18px; text-align:center; padding-top:18px; font-weight:900; font-size:24px; z-index:5; box-sizing:border-box;'>
          Alice<br><span style='font-size:36px;'>TX</span>
        </div>
        <div style='position:absolute; left:50%; transform:translateX(-50%); top:42px; width:150px; height:105px; border:3px solid {eve_border}; background:{eve_color}; border-radius:18px; text-align:center; padding-top:18px; font-weight:900; font-size:24px; display:{eve_display}; z-index:10; box-sizing:border-box;'>
          Eve<br><span style='font-size:20px;'>{eve_label}</span>
        </div>
        <div style='position:absolute; right:2%; top:50px; width:150px; height:90px; border:3px solid #16a34a; background:#dcfce7; border-radius:18px; text-align:center; padding-top:18px; font-weight:900; font-size:24px; z-index:5; box-sizing:border-box;'>
          Bob<br><span style='font-size:36px;'>RX</span>
        </div>
        <div style='position:absolute; left:180px; right:180px; top:96px; height:10px; background:#94a3b8; border-radius:99px; z-index:1;'></div>
        {ball_html}
        <div style='position:absolute; left:2%; bottom:14px; font-size:18px; z-index:5;'>Alice bit = <b>{bit}</b></div>
        <div style='position:absolute; right:2%; bottom:14px; font-size:18px; z-index:5;'>Bob result = <b>{bob_bit}</b></div>
      </div>

      <div style='margin-top:24px;'>
        <div style='font-size:22px; font-weight:800; margin-bottom:8px;'>Alice送信列</div>
        <div style='height:{bit_area_height}px; overflow-y:auto; border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa; margin-bottom:22px;'>
          {bit_list_to_html(alice_bits, index, max_len=total, bits_per_row=bits_per_row, bit_size=bit_size)}
        </div>
        <div style='font-size:22px; font-weight:800; margin-bottom:8px;'>Bob受信列</div>
        <div style='height:{bit_area_height}px; overflow-y:auto; border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa;'>
          {bit_list_to_html(bob_results, index, max_len=total, bits_per_row=bits_per_row, bit_size=bit_size)}
        </div>
      </div>
    </div>
    """
    components.html(html, height=900, scrolling=False)


def animate_bit_transmission(animation_payload, frame_delay=0.04):
    alice_bits = animation_payload["alice_bits"]
    bob_results = animation_payload["bob_results"]
    eve_intervened = animation_payload["eve_intervened"]
    total = min(len(alice_bits), len(bob_results), len(eve_intervened))
    if total == 0:
        return

    st.subheader("0. 量子ビット送信アニメーション")
    st.caption("大きい送信パルス数の場合でも、アニメーションは先頭256パルスまで表示します。")
    placeholder = st.empty()
    phases = [0.0, 0.25, 0.5, 0.75, 1.0]
    for i in range(total):
        for phase in phases:
            placeholder.empty()
            with placeholder.container():
                render_bit_motion_frame(
                    bit=alice_bits[i],
                    index=i,
                    total=total,
                    phase=phase,
                    eve_on=any(x == "○" for x in eve_intervened),
                    eve_hit=eve_intervened[i] == "○",
                    bob_bit=bob_results[i],
                    alice_bits=alice_bits,
                    bob_results=bob_results,
                    show_ball=True,
                )
            time.sleep(frame_delay)

    placeholder.empty()
    with placeholder.container():
        render_bit_motion_frame(
            bit=alice_bits[-1],
            index=total - 1,
            total=total,
            phase=1.0,
            eve_on=any(x == "○" for x in eve_intervened),
            eve_hit=eve_intervened[-1] == "○",
            bob_bit=bob_results[-1],
            alice_bits=alice_bits,
            bob_results=bob_results,
            show_ball=False,
        )


def show_summary_cards(summary, prefix=""):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(f"{prefix}送信", f"{summary['送信パルス数']:,}")
    col2.metric(f"{prefix}検出数", f"{summary['検出数']:,}")
    col3.metric(f"{prefix}鍵候補長", f"{summary['鍵候補長']:,}")
    col4.metric(f"{prefix}QBER", f"{summary['QBER[%]']:.2f}%")
    col5.metric(f"{prefix}最終鍵長", f"{summary['最終鍵長']:,} bit")


if st.button("シミュレーション実行", type="primary"):
    base_seed = int(time.time()) % 1_000_000_000

    bb84_summary, bb84_detail, bb84_anim = simulate_protocol(
        "BB84",
        num_bits,
        mean_photon_mu,
        detection_efficiency,
        dark_count_rate,
        noise_rate,
        eve_enabled,
        eve_rate,
        qber_threshold,
        seed=base_seed,
    )

    t12_summary, t12_detail, t12_anim = simulate_protocol(
        "T12風",
        num_bits,
        mean_photon_mu,
        detection_efficiency,
        dark_count_rate,
        noise_rate,
        eve_enabled,
        eve_rate,
        qber_threshold,
        seed=base_seed + 1,
    )

    if protocol_mode == "従来BB84のみ":
        active_summaries = [bb84_summary]
        active_details = {"BB84": bb84_detail}
        active_anim = bb84_anim
    elif protocol_mode == "T12風のみ":
        active_summaries = [t12_summary]
        active_details = {"T12風": t12_detail}
        active_anim = t12_anim
    else:
        active_summaries = [bb84_summary, t12_summary]
        active_details = {"BB84": bb84_detail, "T12風": t12_detail}
        active_anim = t12_anim

    if show_bit_motion:
        animated_bits = min(num_bits, 256)
        if animated_bits < num_bits:
            st.caption(
                f"送信パルス数は {num_bits:,} ですが、アニメーションは表示負荷を避けるため先頭 {animated_bits} パルスのみ表示しています。"
            )
        animate_bit_transmission(active_anim, frame_delay=0.02)

    st.subheader("1. 全体結果")
    if protocol_mode == "比較表示":
        st.markdown("#### 従来BB84")
        show_summary_cards(bb84_summary)
        st.markdown("#### T12風")
        show_summary_cards(t12_summary)
    else:
        show_summary_cards(active_summaries[0])

    st.subheader("2. BB84とT12風の効率比較")
    compare_df = pd.DataFrame([
        {
            "プロトコル": s["Protocol"],
            "鍵候補効率[%]": s["鍵候補効率[%]"],
            "最終鍵効率[%]": s["最終鍵効率[%]"],
            "QBER[%]": s["QBER[%]"],
            "検出率[%]": s["検出率[%]"],
            "鍵候補長": s["鍵候補長"],
            "最終鍵長": s["最終鍵長"],
        }
        for s in [bb84_summary, t12_summary]
    ])

    c1, c2 = st.columns(2)
    with c1:
        fig_eff = px.bar(
            compare_df,
            x="プロトコル",
            y=["鍵候補効率[%]", "最終鍵効率[%]"],
            barmode="group",
            text_auto=".2f",
            title="送信パルス数に対する鍵生成効率",
        )
        st.plotly_chart(fig_eff, use_container_width=True)
    with c2:
        fig_key = px.bar(
            compare_df,
            x="プロトコル",
            y=["鍵候補長", "最終鍵長"],
            barmode="group",
            text_auto=True,
            title="鍵候補長・最終鍵長の比較",
        )
        st.plotly_chart(fig_key, use_container_width=True)

    st.markdown("#### 比較結果テーブル")
    st.dataframe(compare_df, use_container_width=True)

    t12_gain = t12_summary["鍵候補効率[%]"] / bb84_summary["鍵候補効率[%]"] if bb84_summary["鍵候補効率[%]"] > 0 else 0
    st.info(
        f"今回の条件では、T12風モデルの鍵候補効率は従来BB84の約 {t12_gain:.2f} 倍です。"
        " Z基底を高確率で選ぶため、Z/Z一致が増え、鍵候補として残る割合が高くなります。"
    )

    st.subheader("3. 光子数分布")
    photon_df = pd.DataFrame([
        {"プロトコル": s["Protocol"], "分類": "0光子", "割合[%]": s["0光子率[%]"]} for s in [bb84_summary, t12_summary]
    ] + [
        {"プロトコル": s["Protocol"], "分類": "1光子", "割合[%]": s["1光子率[%]"]} for s in [bb84_summary, t12_summary]
    ] + [
        {"プロトコル": s["Protocol"], "分類": "多光子", "割合[%]": s["多光子率[%]"]} for s in [bb84_summary, t12_summary]
    ])
    fig_photon = px.bar(photon_df, x="分類", y="割合[%]", color="プロトコル", barmode="group", text_auto=".2f", title=f"ポアソン光子数分布 μ={mean_photon_mu:.2f}")
    st.plotly_chart(fig_photon, use_container_width=True)

    st.subheader("4. 計算式・判定ロジック")
    with st.expander("T12風モデルの計算式を見る", expanded=True):
        st.markdown(f"""
        ### 基底選択確率

        従来BB84:

        ```text
        P(Z) = 1/2, P(X) = 1/2
        ```

        T12風:

        ```text
        P(Z) = 15/16 = 0.9375
        P(X) =  1/16 = 0.0625
        ```

        そのため、理想的にはZ/Z一致の確率は以下になります。

        ```text
        従来BB84: P(Z/Z) = 1/2 × 1/2 = 1/4
        T12風   : P(Z/Z) = 15/16 × 15/16 = 225/256 ≒ 87.89%
        ```

        ### 光子数分布

        ```text
        n ~ Poisson(μ)
        ```

        今回の μ は `{mean_photon_mu:.2f}` です。

        ### QBER

        ```text
        QBER = 鍵候補中の誤り数 / 鍵候補長 × 100
        ```

        ### 最終鍵長の簡略モデル

        ```text
        safety_factor = 1 - QBER / QBERしきい値
        final_key_length = int((鍵候補長 - EC leakage) × safety_factor)
        EC leakage = 鍵候補中の誤り数
        ```

        ※ このEC/PAは教育用の簡略モデルです。
        """)

    st.subheader("5. 最終鍵表示")
    for s in active_summaries:
        with st.expander(f"{s['Protocol']} の最終鍵", expanded=False):
            if s["can_generate_key"] and s["最終鍵長"] > 0:
                st.success("鍵生成成功")
                st.code(s["final_key"], language="text")
            else:
                st.error("QBERが高い、または鍵候補が不足しているため、最終鍵は生成されませんでした。")

    if show_detail_table:
        st.subheader("6. 送受信結果の詳細")
        for name, detail in active_details.items():
            with st.expander(f"{name} 詳細表", expanded=(protocol_mode != "比較表示")):
                st.caption(f"詳細表は先頭 {len(detail):,} 行のみ表示しています。計算は全 {num_bits:,} パルスに対して実行しています。")
                st.dataframe(detail, use_container_width=True)
else:
    st.info("左の条件を設定して、［シミュレーション実行］を押してください。")
