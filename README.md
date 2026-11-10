# Flower Federated Learning — Streamlit Demo

Aplikasi demonstrasi Federated Learning untuk konteks data medis dengan satu server, tiga client, dan agregasi FedAvg. Dibuat untuk deployment di Streamlit Community Cloud.

## Menjalankan lokal

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru, misalnya `flower-federated-medical`.
2. Upload seluruh isi folder ini ke root repository (pastikan `app.py` berada di root).
3. Buka https://share.streamlit.io/ dan masuk menggunakan akun GitHub.
4. Pilih **Create app**, pilih repository, branch `main`, dan file utama `app.py`.
5. Klik **Deploy**. Setelah build selesai, Streamlit akan memberikan URL publik `https://....streamlit.app`.

## Catatan metodologi

Aplikasi ini menjalankan simulasi edukatif local update dan agregasi FedAvg menggunakan parameter sintetis. Ini bukan training Mistral-7B dan bukan Flower distributed runtime. Training LLM penuh memerlukan GPU dan resource yang tidak tersedia secara wajar pada Streamlit Community Cloud.
