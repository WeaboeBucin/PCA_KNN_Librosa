import streamlit as st
import pickle
import librosa
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, mode
import scipy.stats as stats

# Judul aplikasi
st.title("Klasifikasi Emosi dari Audio dengan PCA")

# Deskripsi
st.write("Aplikasi ini memungkinkan Anda mengunggah file audio dan melakukan klasifikasi menggunakan PCA dan K-Nearest Neighbors (KNN).")
def hitung_statistik(audio_file):
    zcr_data = pd.DataFrame(columns=['mean', 'std_dev', 'max_value', 'min_value', 'median', 'skewness', 'kurt', 'q1', 'q3', 'mode_value', 'iqr', 'ZCR Mean', 'ZCR Median', 'ZCR Std Deviasi', 'ZCR Skewness', 'ZCR Kurtosis'])
    y, sr = librosa.load(audio_file)
    mean = np.mean(y)
    std_dev = np.std(y)
    max_value = np.max(y)
    min_value = np.min(y)
    median = np.median(y)
    skewness = skew(y)  # Calculate skewness
    kurt = kurtosis(y)  # Calculate kurtosis
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    mode_value, _ = mode(y)  # Calculate mode
    iqr = q3 - q1

    # Hitung ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    # Hitung rata-rata ZCR
    mean_zcr = zcr.mean()
    # Hitung nilai median ZCR
    median_zcr = np.median(zcr)
    # Hitung nilai std deviasa ZCR
    std_dev_zcr = np.std(zcr)
    # Hitung skewness ZCR
    skewness_zcr = stats.skew(zcr, axis=None)
    # Hitung kurtosis ZCR
    kurtosis_zcr = stats.kurtosis(zcr, axis=None)

    # Hitung RMS
    rms = librosa.feature.rms(y=y)
    # Hitung rata-rata RMS
    mean_rms = rms.mean()
    # Hitung nilai median RMS
    median_rms = np.median(rms)
    # Hitung nilai std deviasa RMS
    std_dev_rms = np.std(rms)
    # Hitung skewness RMS
    skewness_rms = stats.skew(rms, axis=None)
    # Hitung kurtosis RMS
    kurtosis_rms = stats.kurtosis(rms, axis=None)

    # Tambahkan data ke DataFrame
    # return[mean, std_dev, max_value, min_value, median, skewness, kurt, q1, q3, mode_value, iqr, mean_zcr, median_zcr, std_dev_zcr, skewness_zcr, kurtosis_zcr, mean_rms, median_rms,std_dev_rms, skewness_rms, kurtosis_rms]
    zcr_data = zcr_data._append({'mean' : mean, 'std_dev' : std_dev, 'max_value' :max_value, 'min_value' :min_value, 'median':median, 'skewness':skewness, 'kurt':kurt, 'q1':q1, 'q3':q3, 'mode_value':mode_value, 'iqr':iqr, 'ZCR Mean': mean_zcr, 'ZCR Median': median_zcr, 'ZCR Std Deviasi': std_dev_zcr, 'ZCR Skewness': skewness_zcr, 'ZCR Kurtosis': kurtosis_zcr,'RMS Mean': mean_rms, 'RMS Median': median_rms, 'RMS Std Deviasi': std_dev_rms, 'RMS Skewness': skewness_rms, 'RMS Kurtosis': kurtosis_rms}, ignore_index=True)
    return zcr_data

with open('standar_scaler.pkl', 'rb') as file:
    standar_scaler = pickle.load(file)

# Memuat model KNN untuk kategori emosi dengan normalisasi ZScore
with open('new_classifier.pkl', 'rb') as file:
    knn_class = pickle.load(file)

with open('new_pca.pkl', 'rb') as file:
    pca_pkl = pickle.load(file)

# Unggah file audio
uploaded_file = st.file_uploader("Unggah 1 file audio (format WAV)", type=["wav"])

if uploaded_file is not None:
    # Memuat data audio (misalnya: fitur audio dari file WAV)
    # Di sini, Anda harus mengganti bagian ini dengan kode yang sesuai untuk membaca dan mengambil fitur-fitur audio.
    # Misalnya, jika Anda menggunakan pustaka librosa, Anda dapat menggunakannya untuk mengambil fitur-fitur audio.
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Deteksi Audio"):
        # audio_data = librosa.load(uploaded_file, sr=None)
        # Simpan file audio yang diunggah
        # audio_path = "audio.wav"
        # with open(audio_path,"wb") as f:
        #     f.write(uploaded_file.getbuffer())
        # Hanya contoh data dummy (harap diganti dengan pengambilan data yang sesungguhnya)
        data_mentah = hitung_statistik(uploaded_file)

        kolom = ['ZCR Mean', 'ZCR Median', 'ZCR Std Deviasi', 'ZCR Kurtosis', 'ZCR Skewness', 'RMS Mean', 'RMS Median', 'RMS Std Deviasi', 'RMS Kurtosis', 'RMS Skewness']
        data_ternormalisasi_zscore = standar_scaler.transform(data_mentah[kolom])

        data_pca = pca_pkl.transform(data_ternormalisasi_zscore)
        label_emosi_pca_standar = knn_class.predict(data_pca)

        # Menampilkan hasil klasifikasi
        st.write("Hasil Klasifikasi Dengan Scaler Standart:")
        st.write(label_emosi_pca_standar)

