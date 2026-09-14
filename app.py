import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load Model, Scaler, dan Features
@st.cache_resource
def load_components():
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('model_features.pkl')
    return model, scaler, features

model, scaler, features = load_components()

# 2. Judul Aplikasi
st.title('🏦 Prediksi Risiko Kredit (Credit Risk Scoring)')
st.write('Final Task ID/X Partners - Data Scientist oleh Firman Adi Ramadhan')
st.markdown('---')

# 3. Form Input dari Pengguna
st.sidebar.header('Masukkan Profil Nasabah')

# Membuat dictionary untuk menampung input pengguna
user_input = {}

# Contoh pembuatan input form (Sesuaikan dengan isi 'final_features' milikmu)
# Karena kamu punya 15 fitur, saya buatkan 4 contoh fundamental. Sisanya bisa kamu tambahkan pola yang sama.
user_input['loan_amnt'] = st.sidebar.number_input('Jumlah Pinjaman (USD)', min_value=500, max_value=35000, value=10000)
user_input['annual_inc'] = st.sidebar.number_input('Pendapatan Tahunan (USD)', min_value=1000, max_value=1000000, value=50000)
user_input['dti'] = st.sidebar.slider('Debt-to-Income Ratio (DTI)', min_value=0.0, max_value=40.0, value=15.0)
user_input['int_rate'] = st.sidebar.slider('Suku Bunga (%)', min_value=5.0, max_value=30.0, value=10.0)
user_input['emp_length_int'] = st.sidebar.slider('Lama Bekerja (Tahun)', min_value=0, max_value=10, value=5)

# PENTING: Untuk sisa 10 fitur lainnya, kamu wajib membuatkan form inputnya di sini juga 
# agar jumlah input sama dengan jumlah fitur di model (15 fitur).
# Berikan nilai default (misal 0) untuk fitur yang bersifat One-Hot Encoding (seperti grade_B, grade_C).

# Mengisi sisa fitur dengan nilai default 0 jika belum diinput oleh form di atas
for col in features:
    if col not in user_input:
        user_input[col] = 0.0  # Nilai default sementara

# 4. Prediksi Tombol
if st.sidebar.button('Prediksi Risiko Kredit'):
    # Mengubah input menjadi DataFrame dengan urutan kolom yang benar
    input_df = pd.DataFrame([user_input])[features]
    
    # Standarisasi data input
    input_scaled = scaler.transform(input_df)
    
    # Memprediksi hasil
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    
    # 5. Menampilkan Hasil
    st.subheader('Hasil Analisis Model:')
    
    if prediction[0] == 1:
        st.error(f'⚠️ BAD CREDIT (Tinggi Risiko Gagal Bayar)')
        st.write(f'Probabilitas Gagal Bayar: **{prediction_proba * 100:.2f}%**')
        st.write('Rekomendasi Bisnis: Tolak pengajuan atau berikan syarat agunan/bunga yang jauh lebih tinggi.')
    else:
        st.success(f'✅ GOOD CREDIT (Aman)')
        st.write(f'Probabilitas Gagal Bayar: **{prediction_proba * 100:.2f}%**')
        st.write('Rekomendasi Bisnis: Pengajuan pinjaman dapat disetujui.')
