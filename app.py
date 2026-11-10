import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Flower Federated Learning | Medical", page_icon="🌼", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px; border-radius: 12px;}
.status-pill {display:inline-block;padding:5px 11px;border-radius:18px;background:#e6f4ea;color:#137333;font-size:13px;font-weight:600;}
.small-note {font-size: .88rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)

st.title("🌼 Flower Federated Learning")
st.subheader("Simulasi Federated Learning untuk Data Medis")
st.markdown("Simulasi satu server agregator dan tiga client lokal menggunakan pendekatan **Federated Averaging (FedAvg)**. Data client disimulasikan secara lokal; data mentah tidak dikirim ke server.")

DEFAULT_CLIENTS = [
    {"client": "Client 01", "dataset": "Medical Flashcards A", "samples": 120, "bias": 0.010},
    {"client": "Client 02", "dataset": "Medical Flashcards B", "samples": 180, "bias": -0.006},
    {"client": "Client 03", "dataset": "Medical Flashcards C", "samples": 150, "bias": 0.003},
]

if "history" not in st.session_state:
    st.session_state.history = []
if "logs" not in st.session_state:
    st.session_state.logs = ["Sistem siap. Tekan Mulai Simulasi untuk menjalankan federasi."]
if "finished" not in st.session_state:
    st.session_state.finished = False
if "rounds_target" not in st.session_state:
    st.session_state.rounds_target = 10

with st.sidebar:
    st.header("⚙️ Pengaturan Simulasi")
    rounds_target = st.slider("Jumlah federated rounds", 3, 20, 10)
    st.session_state.rounds_target = rounds_target
    st.caption("Konfigurasi client")
    for c in DEFAULT_CLIENTS:
        st.write(f"**{c['client']}** · {c['samples']} sampel")
    st.divider()
    st.caption("Strategi agregasi")
    st.code("FedAvg\nweighted average by n_samples", language="text")
    st.caption("Model demonstrasi: vektor parameter numerik sintetis, bukan fine-tuning LLM.")

col1, col2, col3, col4 = st.columns(4)
current_round = len(st.session_state.history)
latest = st.session_state.history[-1] if st.session_state.history else None
col1.metric("Federated Round", f"{current_round}/{st.session_state.rounds_target}")
col2.metric("Global Accuracy", f"{latest['global_accuracy']:.2%}" if latest else "—")
col3.metric("Global Loss", f"{latest['global_loss']:.4f}" if latest else "—")
col4.metric("Jumlah Client", "3")

st.divider()
server_col, clients_col = st.columns([1, 2])
with server_col:
    st.markdown("### 🖥️ Federated Server")
    st.markdown('<span class="status-pill">● READY</span>', unsafe_allow_html=True)
    st.write("**Peran:** menerima parameter lokal dan menjalankan agregasi FedAvg.")
    st.write("**Alamat konseptual:** `server:8080`")
    st.write("**Strategi:** `FedAvg`")
with clients_col:
    st.markdown("### 💻 Federated Clients")
    cc = st.columns(3)
    for box, client in zip(cc, DEFAULT_CLIENTS):
        with box:
            st.markdown(f"**{client['client']}**")
            st.caption(client['dataset'])
            st.metric("Local samples", client["samples"])
            st.markdown('<span class="status-pill">● CONNECTED</span>', unsafe_allow_html=True)

st.divider()
st.markdown("### ▶️ Kontrol Simulasi")
start_col, reset_col, info_col = st.columns([1, 1, 3])
start = start_col.button("▶ Mulai Simulasi", type="primary", use_container_width=True, disabled=st.session_state.finished)
reset = reset_col.button("↻ Reset", use_container_width=True)
if reset:
    st.session_state.history = []
    st.session_state.logs = ["Simulasi direset. Sistem siap menjalankan eksperimen baru."]
    st.session_state.finished = False
    st.rerun()
info_col.info("Satu kali klik menjalankan seluruh round sesuai konfigurasi. Hasil dan log akan ditampilkan setelah proses selesai.")

if start:
    progress = st.progress(0, text="Menginisialisasi federasi...")
    status = st.empty()
    rng = np.random.default_rng(42)
    # Each client starts with a local parameter vector; global model is initialized once.
    global_weights = np.array([0.20, 0.35, 0.45, 0.55, 0.62], dtype=float)
    samples = np.array([c["samples"] for c in DEFAULT_CLIENTS], dtype=float)
    total_samples = samples.sum()
    history = []
    logs = ["Server menginisialisasi global model.", "3 client terdaftar dan siap melakukan local training."]
    base_accuracy = 0.68
    for r in range(1, st.session_state.rounds_target + 1):
        local_weights, local_accs, local_losses = [], [], []
        for idx, client in enumerate(DEFAULT_CLIENTS):
            # Simulated local optimization: each client updates a copy of global parameters.
            noise = rng.normal(0, 0.012, size=global_weights.shape)
            improvement = 0.008 + (r * 0.0015) + client["bias"]
            updated = np.clip(global_weights + improvement + noise, 0.0, 1.0)
            local_weights.append(updated)
            local_accs.append(min(0.96, base_accuracy + r * 0.012 + client["bias"] + rng.normal(0, 0.004)))
            local_losses.append(max(0.08, 0.95 - r * 0.055 + abs(client["bias"]) + rng.normal(0, 0.012)))
            logs.append(f"Round {r}: {client['client']} menyelesaikan local training ({client['samples']} sampel).")
        # FedAvg: weighted mean of local model parameters by each client's sample count.
        global_weights = np.average(np.vstack(local_weights), axis=0, weights=samples)
        global_accuracy = float(np.average(local_accs, weights=samples))
        global_loss = float(np.average(local_losses, weights=samples))
        history.append({"round": r, "global_accuracy": global_accuracy, "global_loss": global_loss,
                        "client_01_accuracy": local_accs[0], "client_02_accuracy": local_accs[1],
                        "client_03_accuracy": local_accs[2], "client_01_loss": local_losses[0],
                        "client_02_loss": local_losses[1], "client_03_loss": local_losses[2]})
        logs.append(f"Round {r}: server melakukan agregasi FedAvg; global accuracy {global_accuracy:.2%}.")
        progress.progress(r / st.session_state.rounds_target, text=f"Memproses federated round {r}/{st.session_state.rounds_target}")
        status.write(f"Round {r} selesai · agregasi global berhasil.")
        time.sleep(0.12)
    st.session_state.history = history
    st.session_state.logs = logs + [f"Simulasi selesai: {st.session_state.rounds_target} round berhasil dijalankan."]
    st.session_state.finished = True
    st.rerun()

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.divider()
    st.markdown("### 📈 Hasil Training")
    chart1, chart2 = st.columns(2)
    with chart1:
        fig = px.line(df, x="round", y="global_accuracy", markers=True, title="Global Model Accuracy", labels={"round":"Federated Round", "global_accuracy":"Accuracy"})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with chart2:
        fig = px.line(df, x="round", y="global_loss", markers=True, title="Global Model Loss", labels={"round":"Federated Round", "global_loss":"Loss"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Performa Setiap Client")
    acc_df = df.melt(id_vars="round", value_vars=["client_01_accuracy", "client_02_accuracy", "client_03_accuracy"], var_name="client", value_name="accuracy")
    fig = px.line(acc_df, x="round", y="accuracy", color="client", markers=True, title="Local Accuracy per Client")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ringkasan Hasil")
    first, last = df.iloc[0], df.iloc[-1]
    a, b, c = st.columns(3)
    a.metric("Accuracy awal", f"{first['global_accuracy']:.2%}")
    b.metric("Accuracy akhir", f"{last['global_accuracy']:.2%}", delta=f"{(last['global_accuracy']-first['global_accuracy']):.2%}")
    c.metric("Penurunan loss", f"{(first['global_loss']-last['global_loss']):.4f}")
    st.dataframe(df.round(4), use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh hasil simulasi (CSV)", data=csv, file_name="flower_federated_simulation_results.csv", mime="text/csv")

st.divider()
st.markdown("### 🧾 Federated Event Log")
for line in st.session_state.logs[-30:]:
    st.write("• " + line)

with st.expander("Metodologi dan batasan simulasi"):
    st.markdown("""
    **Alur eksperimen**
    1. Server menginisialisasi parameter global.
    2. Tiga client menerima parameter global dan melakukan local update pada data lokal sintetis.
    3. Client mengirim parameter model hasil update (bukan data mentah) ke server.
    4. Server mengagregasikan parameter menggunakan FedAvg: setiap parameter global merupakan rata-rata tertimbang berdasarkan jumlah sampel client.
    5. Proses diulang sesuai jumlah federated round.

    **Rumus FedAvg**

    `w_global = Σ (n_k / N) × w_k`

    dengan `n_k` jumlah sampel client ke-k, `N` total sampel seluruh client, dan `w_k` parameter model lokal client ke-k.

    **Batasan:** aplikasi ini adalah demonstrasi interaktif mekanisme federated learning dan agregasi FedAvg dengan parameter serta metrik sintetis. Aplikasi tidak menjalankan training Mistral-7B, tidak memuat dataset medis asli, dan tidak membuka koneksi Flower server-client terdistribusi. Untuk eksperimen LLM penuh, gunakan lingkungan komputasi GPU terpisah.
    """)

st.caption("FlowerTune LLM Medical · Streamlit demonstration · Educational simulation")
