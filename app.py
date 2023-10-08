import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pickle import dump
from sklearn.preprocessing import StandardScaler
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import skew, kurtosis, mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Judul aplikasi
st.title("Klasifikasi Audio dengan PCA")

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
    zcr_data = zcr_data.append({'mean' : mean, 'std_dev' : std_dev, 'max_value' :max_value, 'min_value' :min_value, 'median':median, 'skewness':skewness, 'kurt':kurt, 'q1':q1, 'q3':q3, 'mode_value':mode_value, 'iqr':iqr, 'ZCR Mean': mean_zcr, 'ZCR Median': median_zcr, 'ZCR Std Deviasi': std_dev_zcr, 'ZCR Skewness': skewness_zcr, 'ZCR Kurtosis': kurtosis_zcr,'RMS Mean': mean_rms, 'RMS Median': median_rms, 'RMS Std Deviasi': std_dev_rms, 'RMS Skewness': skewness_rms, 'RMS Kurtosis': kurtosis_rms}, ignore_index=True)
    return zcr_data


dataknn= pd.read_csv('hasil_zcr_rms.csv')
X = dataknn.drop(['Label','File'], axis=1)  # Ganti 'target_column' dengan nama kolom target
y = dataknn['Label']
# split data into train and test sets
X_train,X_test,y_train, y_test= train_test_split(X, y, random_state=1, test_size=0.2)

# Unggah file audio
uploaded_file = st.file_uploader("Unggah file audio (format WAV)", type=["wav"])

if uploaded_file is not None:
    # Memuat data audio (misalnya: fitur audio dari file WAV)
    # Di sini, Anda harus mengganti bagian ini dengan kode yang sesuai untuk membaca dan mengambil fitur-fitur audio.
    # Misalnya, jika Anda menggunakan pustaka librosa, Anda dapat menggunakannya untuk mengambil fitur-fitur audio.
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Deteksi Audio"):
        # Simpan file audio yang diunggah
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Hanya contoh data dummy (harap diganti dengan pengambilan data yang sesungguhnya)
        data_mentah = hitung_statistik(audio_path)
        data_mentah.to_csv('hasil_zcr_rms1.csv', index=False)
        # Standarisasi fitur (opsional, tapi dapat meningkatkan kinerja PCA dan KNN)
        kolom = ['ZCR Mean', 'ZCR Median', 'ZCR Std Deviasi', 'ZCR Kurtosis', 'ZCR Skewness', 'RMS Mean', 'RMS Median', 'RMS Std Deviasi', 'RMS Kurtosis', 'RMS Skewness']
        # Inisialisasi StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        # Lakukan standarisasi pada kolom yang telah ditentukan
        data_ternormalisasi = scaler.transform(data_mentah)

        MinimMaximscaler = MinMaxScaler(feature_range=(0, 1))
        MinimMaximscaler.fit(X_train)
        data_MinimMaxim = MinimMaximscaler.transform(data_mentah)
        # Reduksi dimensi menggunakan PCA
        sklearn_pca = PCA(n_components=1)
        X_train_pca = sklearn_pca.fit_transform(X_train_scaled)
        X_pca = sklearn_pca.transform(data_ternormalisasi)
        MM_pca = sklearn_pca.transform(data_MinimMaxim)

        # Inisialisasi KNN Classifier
        classifier = KNeighborsClassifier(n_neighbors=2)
        classifier.fit(X_train_pca, y_train)
        y_prediksi = classifier.predict(X_pca)

        # Menampilkan hasil klasifikasi
        st.write("Hasil Klasifikasi:")
        st.write(y_prediksi)
