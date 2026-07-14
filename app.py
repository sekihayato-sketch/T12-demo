import streamlit as st
import random
import pandas as pd
import hashlib
import time
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(page_title="BB84 QKD Simulator", layout="wide")

st.title("BB84量子鍵配送シミュレータ")
st.caption("シフティング -> QBER評価 -> 誤り訂正 -> 秘匿性増幅までを体験する教育用アプリ")

st.markdown("""
このアプリでは、BB84プロトコルにおける **基底選択**、**シフティング**、**QBERによる鍵破棄判定**、
簡略化した **誤り訂正** と **秘匿性増幅** を確認できます。

※ 誤り訂正と秘匿性増幅は、インターン説明用の簡略モデルです。実際のQKD装置で使う実装とは異なります。
""")

with st.expander("通信ノイズについて", expanded=False):
    st.markdown("""
    このアプリでは、Eveによる盗聴とは別に **通信ノイズ** も設定できます。

    通信ノイズは、量子チャネルや受信系の揺らぎなどにより、Bobの測定結果が誤る効果を簡略的に表しています。

    このアプリでは、Bobが測定した後に、指定した確率でビットを反転させています。

    ```text
    Bob Result = 0 の場合、ノイズ発生で 1 に反転
    Bob Result = 1 の場合、ノイズ発生で 0 に反転
    ```

    そのため、Eveが存在しない場合でも、通信ノイズ率を上げるとQBERが増加します。

    ### Eveと通信ノイズの違い

    - **Eveによる盗聴**：量子状態を途中で測定することで、基底不一致時に状態を乱す
    - **通信ノイズ**：盗聴者がいなくても、伝送路や測定系の揺らぎによりBobの結果が誤る

    実際のQKDでは、QBERには盗聴だけでなく、光学系の不完全性、検出器ノイズ、アライメントずれ、伝送損失なども影響します。
    """)

with st.sidebar:
    st.header("シミュレーション条件")
    num_bits = st.select_slider(
        "送信ビット数",
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
            1048576
        ],
        value=4096
    )
    
    eve_enabled = st.checkbox("Eveによる盗聴を有効化", value=True)
    eve_rate = st.slider("Eveが介入する割合 [%]", 0, 100, 100, step=5, disabled=not eve_enabled)
    noise_rate = st.slider("通信路ノイズ率 [%]", 0, 20, 0, step=1)
    qber_threshold = st.slider("鍵破棄しきい値 QBER [%]", 0.0, 20.0, 11.0, step=0.5)
    animation_speed = st.slider("アニメーション速度", 0.1, 1.5, 0.6, step=0.1)
    show_animation = st.checkbox("処理の流れをアニメーション表示", value=True)
    show_bit_motion = st.checkbox("0/1ビットの送信アニメーションを表示", value=True)

max_animated_bits = st.slider(
    "送信アニメーションで表示するビット数",
    1,
    32,
    12,
    step=1
)

bases = ["+", "×"]


def bits_to_string(bits):
    return "".join(str(b) for b in bits) if bits else "-"


def short_bits(bits, max_len=64):
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


def render_flow(active_step, status_text):
    steps = [
        ("Alice", "ビット・基底生成"),
        ("量子チャネル", "量子状態を送信"),
        ("Eve", "盗聴・測定"),
        ("Bob", "ランダム基底で測定"),
        ("Sifting", "基底一致だけ採用"),
        ("QBER", "誤り率を評価"),
        ("EC", "誤り訂正"),
        ("PA", "秘匿性増幅"),
        ("Final Key", "最終鍵生成"),
    ]

    st.markdown("#### BB84処理の流れ")
    cols = st.columns(len(steps))
    for idx, (title, subtitle) in enumerate(steps):
        if idx < active_step:
            color = "#d9f2e6"
            border = "#20a060"
            mark = "OK"
        elif idx == active_step:
            color = "#fff3cd"
            border = "#f0ad00"
            mark = "NOW"
        else:
            color = "#f3f4f6"
            border = "#c7c7c7"
            mark = "WAIT"

        with cols[idx]:
            st.markdown(
                f"""
                <div style='border:2px solid {border}; background:{color}; border-radius:12px; padding:10px; text-align:center; min-height:110px;'>
                    <div style='font-size:13px; font-weight:bold;'>{mark}</div>
                    <div style='font-size:20px; font-weight:bold; margin-top:5px;'>{title}</div>
                    <div style='font-size:12px; margin-top:6px;'>{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.info(status_text)

def bit_list_to_html(bits, current_index=None, max_len=None, bits_per_row=32, bit_size=14):
    if max_len is None:
        display_bits = bits
    else:
        display_bits = bits[:max_len]

    cell_width = bit_size + 28
    cell_height = bit_size + 30

    html = (
        f"<div style='display:grid; "
        f"grid-template-columns:repeat({bits_per_row}, {cell_width}px); "
        f"gap:6px; align-items:center;'>"
    )

    for i, bit in enumerate(display_bits):
        active = i == current_index
        bg = "#2563eb" if str(bit) == "1" else "#0f766e"
        border = "4px solid #facc15" if active else "1px solid #d1d5db"

        html += (
            f"<div style='width:{cell_width}px; height:{cell_height}px; "
            f"border-radius:8px; background:{bg}; color:white; border:{border}; "
            f"display:flex; align-items:center; justify-content:center; "
            f"font-weight:900; font-size:{bit_size}px; box-sizing:border-box;'>"
            f"{bit}</div>"
        )

    html += "</div>"
    return html

def render_bit_motion_frame(
    bit,
    index,
    total,
    phase,
    eve_on,
    eve_hit,
    bob_bit,
    alice_bits,
    bob_results,
    show_ball=True
):
    left = 5 + phase * 78

    eve_display = "block" if eve_on else "none"
    eve_color = "#fee2e2" if eve_hit else "#f3f4f6"
    eve_border = "#dc2626" if eve_hit else "#9ca3af"
    eve_label = "Eve測定" if eve_hit else "Eve待機"

    # 256 bitでも破綻しない表示設定
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

    component_height = 900

    ball_html = ""
    if show_ball:
        ball_html = f"""
        <div style='position:absolute; left:{left}%; top:68px; width:54px; height:54px;
                    border-radius:50%; background:#111827; color:#ffffff;
                    display:flex; align-items:center; justify-content:center;
                    font-size:30px; font-weight:900;
                    box-shadow:0 0 18px rgba(37,99,235,0.7);
                    z-index:8;'>
          {bit}
        </div>
        """

    html = f"""
    <div style='border:1px solid #d1d5db; border-radius:16px;
                padding:18px; background:#ffffff; box-sizing:border-box;'>

      <div style='font-weight:700; margin-bottom:10px; font-size:20px;'>
        送信ビット {index + 1} / {total}：AliceからBobへ量子状態を送信中
      </div>

      <div style='position:relative; height:210px;
                  background:linear-gradient(90deg,#eff6ff,#ffffff,#f0fdf4);
                  border-radius:14px; overflow:hidden;'>

        <div style='position:absolute; left:2%; top:50px; width:150px; height:90px;
                    border:3px solid #2563eb; background:#dbeafe; border-radius:18px;
                    text-align:center; padding-top:18px; font-weight:900;
                    font-size:24px; z-index:5; box-sizing:border-box;'>
          Alice<br><span style='font-size:36px;'>TX</span>
        </div>

        <div style='position:absolute; left:50%; transform:translateX(-50%);
                    top:42px; width:150px; height:105px;
                    border:3px solid {eve_border}; background:{eve_color}; border-radius:18px;
                    text-align:center; padding-top:18px; font-weight:900;
                    font-size:24px; display:{eve_display}; z-index:10; box-sizing:border-box;'>
          Eve<br><span style='font-size:20px;'>{eve_label}</span>
        </div>

        <div style='position:absolute; right:2%; top:50px; width:150px; height:90px;
                    border:3px solid #16a34a; background:#dcfce7; border-radius:18px;
                    text-align:center; padding-top:18px; font-weight:900;
                    font-size:24px; z-index:5; box-sizing:border-box;'>
          Bob<br><span style='font-size:36px;'>RX</span>
        </div>

        <div style='position:absolute; left:180px; right:180px; top:96px;
                    height:10px; background:#94a3b8; border-radius:99px;
                    z-index:1;'>
        </div>

        {ball_html}

        <div style='position:absolute; left:2%; bottom:14px; font-size:18px; z-index:5;'>
          Alice bit = <b>{bit}</b>
        </div>

        <div style='position:absolute; right:2%; bottom:14px; font-size:18px; z-index:5;'>
          Bob result = <b>{bob_bit}</b>
        </div>
      </div>

      <div style='margin-top:24px;'>

        <div style='font-size:22px; font-weight:800; margin-bottom:8px;'>
          Alice送信列
        </div>

        <div style='height:{bit_area_height}px; overflow-y:auto;
                    border:1px solid #e5e7eb; border-radius:12px;
                    padding:12px; background:#fafafa; margin-bottom:22px;'>
          {bit_list_to_html(
              alice_bits,
              index,
              max_len=total,
              bits_per_row=bits_per_row,
              bit_size=bit_size
          )}
        </div>

        <div style='font-size:22px; font-weight:800; margin-bottom:8px;'>
          Bob受信列
        </div>

        <div style='height:{bit_area_height}px; overflow-y:auto;
                    border:1px solid #e5e7eb; border-radius:12px;
                    padding:12px; background:#fafafa;'>
          {bit_list_to_html(
              bob_results,
              index,
              max_len=total,
              bits_per_row=bits_per_row,
              bit_size=bit_size
          )}
        </div>

      </div>

    </div>
    """

    components.html(html, height=component_height, scrolling=False)
    
def animate_bit_transmission(
    alice_bits,
    bob_results,
    eve_intervened,
    max_bits=12,
    frame_delay=0.08
):
    st.subheader("0. 量子ビット送信アニメーション")
    st.caption("AliceからBobへ、0/1の量子状態が順番に送られる様子を簡略表示します。")

    total = min(max_bits, len(alice_bits), len(bob_results))

    display_alice_bits = alice_bits[:total]
    display_bob_results = bob_results[:total]
    display_eve_intervened = eve_intervened[:total]

    placeholder = st.empty()

    for i in range(total):
        for phase in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            placeholder.empty()
            with placeholder.container():
                render_bit_motion_frame(
                    bit=display_alice_bits[i],
                    index=i,
                    total=total,
                    phase=phase,
                    eve_on=any(x == "○" for x in display_eve_intervened),
                    eve_hit=display_eve_intervened[i] == "○",
                    bob_bit=display_bob_results[i],
                    alice_bits=display_alice_bits,
                    bob_results=display_bob_results,
                    show_ball=True
                )

            time.sleep(frame_delay)

    # 最終フレーム：最後のビットをハイライトし、移動中のボールだけ消す
    placeholder.empty()
    with placeholder.container():
        render_bit_motion_frame(
            bit=display_alice_bits[-1],
            index=total - 1,
            total=total,
            phase=1.0,
            eve_on=any(x == "○" for x in display_eve_intervened),
            eve_hit=display_eve_intervened[-1] == "○",
            bob_bit=display_bob_results[-1],
            alice_bits=display_alice_bits,
            bob_results=display_bob_results,
            show_ball=False
        )
        
def run_animation(alice_bits, alice_key, bob_key, corrected_bob_key, final_key, qber, can_generate_key):
    placeholder = st.empty()
    animation_steps = [
        (0, "Aliceがランダムなビット列と送信用基底を生成しています。"),
        (1, "Aliceが量子状態を量子チャネルへ送信しています。"),
        (2, "Eveが設定条件に従って一部または全部の量子状態を測定します。基底が違うと状態が乱れる可能性があります。"),
        (3, "Bobがランダムに選んだ基底で測定しています。"),
        (4, "AliceとBobが基底情報だけを公開し、一致した位置だけを鍵候補に残します。"),
        (5, f"鍵候補の誤り率 QBER を評価しています。今回のQBERは {qber:.2f}% です。"),
        (6, "QBERがしきい値以下の場合、簡略化した誤り訂正でBob側の鍵候補をAlice側にそろえます。"),
        (7, "誤り訂正で漏れた情報を考慮し、秘匿性増幅で鍵を短く圧縮します。"),
        (8, "最終鍵生成結果を表示します。"),
    ]

    progress = st.progress(0)
    for index, (step_no, message) in enumerate(animation_steps):
        with placeholder.container():
            render_flow(step_no, message)
            if step_no == 0:
                st.code("Alice送信ビット: " + short_bits(alice_bits), language="text")
            elif step_no == 4:
                st.code("Alice鍵候補: " + short_bits(alice_key), language="text")
                st.code("Bob鍵候補  : " + short_bits(bob_key), language="text")
            elif step_no == 6:
                st.code("誤り訂正後 Bob鍵候補: " + short_bits(corrected_bob_key), language="text")
            elif step_no == 8:
                if can_generate_key and final_key != "-":
                    st.success("最終鍵生成成功")
                    st.code("Final Key: " + final_key, language="text")
                else:
                    st.error("最終鍵は生成されませんでした。鍵候補を破棄します。")
        progress.progress((index + 1) / len(animation_steps))
        time.sleep(animation_speed)


if st.button("シミュレーション実行", type="primary"):
    alice_bits = [random.randint(0, 1) for _ in range(num_bits)]
    alice_bases = [random.choice(bases) for _ in range(num_bits)]
    bob_bases = [random.choice(bases) for _ in range(num_bits)]

    bob_results = []
    eve_bases = []
    eve_results = []
    eve_intervened = []
    noise_flipped = []

    for i in range(num_bits):
        transmitted_bit = alice_bits[i]
        transmitted_basis = alice_bases[i]

        does_eve_intervene = eve_enabled and random.randint(1, 100) <= eve_rate
        eve_intervened.append("○" if does_eve_intervene else "-")

        if does_eve_intervene:
            eve_basis = random.choice(bases)
            eve_bases.append(eve_basis)

            if eve_basis == alice_bases[i]:
                eve_result = alice_bits[i]
            else:
                eve_result = random.randint(0, 1)

            eve_results.append(eve_result)
            transmitted_bit = eve_result
            transmitted_basis = eve_basis
        else:
            eve_bases.append("-")
            eve_results.append("-")

        if bob_bases[i] == transmitted_basis:
            bob_result = transmitted_bit
        else:
            bob_result = random.randint(0, 1)

        does_noise_flip = random.randint(1, 100) <= noise_rate
        if does_noise_flip:
            bob_result = 1 - bob_result
        noise_flipped.append("○" if does_noise_flip else "-")

        bob_results.append(bob_result)

    same_basis = [alice_bases[i] == bob_bases[i] for i in range(num_bits)]
    used_for_key = same_basis
    key_error = [same_basis[i] and alice_bits[i] != bob_results[i] for i in range(num_bits)]

    alice_key = [alice_bits[i] for i in range(num_bits) if used_for_key[i]]
    bob_key = [bob_results[i] for i in range(num_bits) if used_for_key[i]]

    key_length = len(alice_key)
    errors = sum(1 for a, b in zip(alice_key, bob_key) if a != b)
    qber = errors / key_length * 100 if key_length > 0 else 0.0

    can_generate_key = key_length > 0 and qber <= qber_threshold

    if can_generate_key:
        corrected_bob_key = alice_key.copy()
        ec_leakage_bits = errors
    else:
        corrected_bob_key = bob_key.copy()
        ec_leakage_bits = 0

    safety_factor = max(0.0, 1.0 - qber / qber_threshold) if qber_threshold > 0 else 0.0

    if can_generate_key:
        final_key_length = max(0, int((key_length - ec_leakage_bits) * safety_factor))
        final_key = hash_to_bits(bits_to_string(corrected_bob_key), final_key_length)
    else:
        final_key_length = 0
        final_key = "-"

    if show_bit_motion:
        animated_bits = min(num_bits, 256)

        if animated_bits < num_bits:
            st.caption(
                f"送信ビット数は {num_bits:,} bit ですが、"
                f"アニメーションは表示負荷を避けるため先頭 {animated_bits} bit のみ表示しています。"
            )

        animate_bit_transmission(
            alice_bits=alice_bits,
            bob_results=bob_results,
            eve_intervened=eve_intervened,
            max_bits=animated_bits,
            frame_delay=0.08
        )
    
    if show_animation:
        run_animation(alice_bits, alice_key, bob_key, corrected_bob_key, final_key, qber, can_generate_key)

    st.subheader("1. 全体結果")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("送信ビット数", num_bits)
    col2.metric("基底一致数", key_length)
    col3.metric("誤り数", errors)
    col4.metric("QBER", f"{qber:.2f}%")
    col5.metric("最終鍵長", f"{final_key_length} bit")

    st.write("QBERしきい値に対する現在値")
    progress_value = min(qber / qber_threshold, 1.0) if qber_threshold > 0 else 1.0
    st.progress(progress_value)

    if can_generate_key and final_key_length > 0:
        st.success("QBERがしきい値以下です。誤り訂正と秘匿性増幅を行い、最終鍵を生成できます。")
    elif can_generate_key and final_key_length == 0:
        st.warning("QBERはしきい値以下ですが、秘匿性増幅後の鍵長が0 bitになりました。送信ビット数を増やすか、誤りを減らしてください。")
    else:
        st.error("QBERがしきい値を超えています。安全な鍵生成は行わず、鍵候補を破棄します。")

    st.subheader("2. BB84の処理フロー")
    f1, f2, f3, f4 = st.columns(4)
    f1.info("① 量子送信\n\nAliceがビットと基底をランダムに選んで送信")
    f2.info("② シフティング\n\nAliceとBobの基底が一致した位置だけ鍵候補に採用")
    f3.info("③ QBER評価\n\n鍵候補のAlice/Bob不一致率を計算")
    f4.info("④ EC・PA\n\n誤り訂正後、秘匿性増幅で鍵を短く圧縮")

    st.subheader("2.5 誤り訂正・秘匿性増幅の計算式")

    with st.expander("計算式の説明を表示する", expanded=True):
        st.markdown(f"""
        このアプリでは、インターン説明用に以下の **簡略モデル** で誤り訂正と秘匿性増幅を表現しています。

        ### 1. QBERの計算

        QBERは、シフティング後の鍵候補のうち、AliceとBobで値が一致しなかった割合です。

        ```text
        QBER [%] = 誤り数 / 基底一致数 × 100
        ```

        今回の値は、

        ```text
        QBER = {errors} / {key_length} × 100 = {qber:.2f} %
        ```

        ### 2. 誤り訂正 EC: Error Correction

        実際のQKDでは、AliceとBobが一部の情報を公開しながら、鍵候補の不一致を訂正します。

        このアプリでは教育用に、次のように単純化しています。

        ```text
        誤り訂正後のBob鍵候補 = Alice鍵候補に一致させる
        ```

        また、誤り訂正の過程で公開された情報量を、簡略的に次のように置いています。

        ```text
        EC leakage [bit] = 誤り数
        ```

        今回の値は、

        ```text
        EC leakage = {errors} bit
        ```

        ### 3. 秘匿性増幅 PA: Privacy Amplification

        誤り訂正で一部の情報が漏れた可能性があるため、最終鍵は短く圧縮します。

        このアプリでは、QBERが高いほど鍵を強く短縮する簡略モデルにしています。

        ```text
        safety_factor = 1 - QBER / QBERしきい値
        ```

        今回の値は、

        ```text
        safety_factor = 1 - {qber:.2f} / {qber_threshold:.2f}
        ```

        最終鍵長は次の式で計算しています。

        ```text
        最終鍵長 [bit] = int((基底一致数 - EC leakage) × safety_factor)
        ```

        今回の値は、

        ```text
        最終鍵長 = int(({key_length} - {errors}) × {safety_factor:.3f})
                 = {final_key_length} bit
        ```

        ### 注意

        このEC/PAは、理解しやすさを優先した簡略モデルです。  
        実際のQKD装置では、Cascade、LDPC、Toeplitz hashingなど、より厳密な誤り訂正・秘匿性増幅処理が使われます。
        """)
    
    st.subheader("3. 鍵の変化")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**シフティング後のAlice鍵候補**")
        st.code(bits_to_string(alice_key), language="text")
        st.markdown("**シフティング後のBob鍵候補**")
        st.code(bits_to_string(bob_key), language="text")
    with k2:
        st.markdown("**誤り訂正後のBob鍵候補（簡略モデル）**")
        st.code(bits_to_string(corrected_bob_key), language="text")
        st.markdown("**秘匿性増幅後の最終鍵（簡略モデル）**")
        st.code(final_key, language="text")

    st.subheader("4. 視覚的な内訳")

    summary_df = pd.DataFrame({
        "項目": ["送信ビット", "基底一致", "鍵候補中の誤り", "最終鍵"],
        "bit数": [num_bits, key_length, errors, final_key_length],
    })

    fig = px.bar(
        summary_df,
        x="項目",
        y="bit数",
        color="項目",
        text="bit数",
        category_orders={"項目": ["送信ビット", "基底一致", "鍵候補中の誤り", "最終鍵"]},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="bit数", xaxis_title="")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("5. 送受信結果の詳細")
    df = pd.DataFrame({
        "No.": list(range(1, num_bits + 1)),
        "Alice Bit": alice_bits,
        "Alice Basis": alice_bases,
        "Eve介入": eve_intervened,
        "Eve Basis": eve_bases,
        "Eve Result": eve_results,
        "Bob Basis": bob_bases,
        "Bob Result": bob_results,
        "Noise Flip": noise_flipped,
        "Basis Match": ["○" if x else "×" for x in same_basis],
        "Used for Key": ["○" if x else "×" for x in used_for_key],
        "Key Error": ["○" if x else "-" for x in key_error],
        "意味": [
            "鍵候補だが誤りあり" if key_error[i]
            else "鍵候補" if used_for_key[i]
            else "基底不一致で破棄"
            for i in range(num_bits)
        ],
    })

    def highlight_rows(row):
        if row["Key Error"] == "○":
            return ["background-color: #ffe6e6"] * len(row)
        if row["Used for Key"] == "○":
            return ["background-color: #e8f5e9"] * len(row)
        return ["background-color: #f5f5f5"] * len(row)

    max_display_rows = 500

    df_display = df.head(max_display_rows)

    st.caption(
        f"詳細表は先頭 {max_display_rows} 行のみ表示しています。"
        f" 計算は全 {num_bits:,} bit に対して実行しています。"
    )

    st.dataframe(df_display.style.apply(highlight_rows, axis=1), use_container_width=True)

    with st.expander("この表の見方"):
        st.markdown("""
        - **Basis Match = ○**：AliceとBobの基底が一致した位置です。シフティングで鍵候補になります。
        - **Used for Key = ○**：シフティング後の鍵候補に採用された位置です。
        - **Key Error = ○**：鍵候補に採用されたが、Alice BitとBob Resultが一致していない位置です。
        - **QBER**：`Key Errorの数 ÷ 基底一致数 × 100` で計算しています。
        - **Eve介入 = ○** でも、直接その行を破棄しているわけではありません。破棄判断はQBERで行います。
        """)
else:
    st.info("左の条件を設定して、［シミュレーション実行］を押してください。")
