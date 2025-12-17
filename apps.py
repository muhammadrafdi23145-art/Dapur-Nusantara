import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import uuid
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Chatbot Dapur Nusantara",
    layout="centered"
)

# =========================
# SESSION STATE
# =========================
if "keranjang" not in st.session_state:
    st.session_state.keranjang = []

# =========================
# DATA UMKM
# =========================
UMKM_NAME = "Dapur Nusantara"
UMKM_DESC = "UMKM makanan rumahan yang menyediakan menu khas Nusantara dengan harga terjangkau."

menu_makanan = {
    "makanan_berat": [
        {"nama": "Nasi Goreng Kampung", "harga": 18000, "rasa": ["gurih"]},
        {"nama": "Ayam Geprek", "harga": 20000, "rasa": ["pedas"]},
        {"nama": "Mie Goreng Jawa", "harga": 17000, "rasa": ["gurih"]},
    ],
    "cemilan": [
        {"nama": "Tahu Crispy", "harga": 10000, "rasa": ["gurih"]},
        {"nama": "Pisang Goreng", "harga": 10000, "rasa": ["manis"]},
    ],
    "minuman": [
        {"nama": "Es Teh", "harga": 5000, "rasa": ["manis"]},
        {"nama": "Es Jeruk", "harga": 7000, "rasa": ["manis"]},
    ]
}

# =========================
# FAQ UMKM
# =========================
faq_pairs = [
    {"q": "menu makanan", "a": "Kami menyediakan nasi goreng, ayam geprek, mie goreng, cemilan, dan minuman."},
    {"q": "harga", "a": "Harga menu kami mulai dari Rp5.000 sampai Rp20.000."},
    {"q": "jam buka", "a": "UMKM buka setiap hari pukul 09.00 – 21.00."},
    {"q": "pembayaran", "a": "Pembayaran bisa tunai dan transfer."},
    {"q": "lokasi", "a": "Kami melayani pesan antar dan ambil di tempat."},
    {"q": "halal", "a": "Semua menu UMKM kami 100% halal."},
]

# =========================
# MODEL NLP
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()
faq_q = [f["q"] for f in faq_pairs]
faq_emb = model.encode(faq_q)

# =========================
# HELPER
# =========================
def cari_menu(nama):
    for kategori in menu_makanan.values():
        for item in kategori:
            if nama in item["nama"].lower():
                return item
    return None

def lihat_keranjang():
    if not st.session_state.keranjang:
        return "Keranjang masih kosong."

    total = 0
    hasil = ""
    for i, item in enumerate(st.session_state.keranjang, 1):
        hasil += f"{i}. {item['nama']} x{item['jumlah']} = Rp{item['total']}\n"
        total += item["total"]

    hasil += f"\nTOTAL: Rp{total}"
    return hasil

def rekomendasi_makanan(rasa):
    hasil = []
    for kategori, daftar in menu_makanan.items():
        for item in daftar:
            if rasa in item["rasa"]:
                hasil.append(f"- {item['nama']} (Rp{item['harga']})")
    return "\n".join(hasil) if hasil else "Belum ada menu yang cocok dengan selera tersebut."

def checkout():
    if not st.session_state.keranjang:
        return "Keranjang masih kosong."

    total = sum(item["total"] for item in st.session_state.keranjang)
    order_id = str(uuid.uuid4())[:8].upper()
    waktu = datetime.now().strftime("%d-%m-%Y %H:%M")

    st.session_state.keranjang.clear()

    return f"""
✅ **Pesanan Berhasil!**

UMKM : {UMKM_NAME}  
Order ID : {order_id}  
Waktu : {waktu}  
Total Bayar : Rp{total}

Terima kasih sudah mendukung Dapur Nusantara 🙏
"""

def chatbot(pesan):
    pesan = pesan.lower()

    if "keranjang" in pesan:
        return lihat_keranjang()

    if "checkout" in pesan:
        return checkout()

    if "rekomendasi" in pesan:
        rasa = pesan.replace("rekomendasi", "").strip()
        return rekomendasi_makanan(rasa)

    # Pesan makanan
    match = re.search(r"(pesan|order)\s+(.+?)\s+(\d+)", pesan)
    if match:
        nama = match.group(2)
        jumlah = int(match.group(3))
        item = cari_menu(nama)
        if item:
            total = item["harga"] * jumlah
            st.session_state.keranjang.append({
                "nama": item["nama"],
                "jumlah": jumlah,
                "total": total
            })
            return f"{item['nama']} x{jumlah} berhasil ditambahkan ke keranjang."
        return "Menu tidak ditemukan."

    # FAQ semantic
    emb = model.encode([pesan])
    sim = np.dot(emb, faq_emb.T)
    idx = np.argmax(sim)
    return faq_pairs[idx]["a"]

# =========================
# UI
# =========================
st.title("Chatbot Dapur Nusantara")
st.write(f"**{UMKM_NAME}** – {UMKM_DESC}")

user_input = st.text_input("Ketik pesan (contoh: pesan ayam geprek 2)")

if st.button("Kirim") and user_input:
    jawaban = chatbot(user_input)
    st.markdown("### Balasan")
    st.markdown(jawaban)
