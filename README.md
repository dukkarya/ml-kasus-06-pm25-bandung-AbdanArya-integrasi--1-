# Prakiraan PM2.5 Bandung

**Machine Learning Project — Kasus 06 | MKB UPI**  
**Tim: Rizqi Abdan_2405122 & Mochammad Arya Lukmanul Hakim_2411157

| Informasi | Keterangan |
|---|---|
| Kasus | Kasus 06 — Prakiraan PM2.5 Bandung |
| Bidang | Supervised Learning — Time-Series Regression |
| Aplikasi | [Buka Aplikasi Prakiraan PM2.5](https://ml-kasus-06-pm-uzs2bl.streamlit.app/) |
| Repository | [Repository: https://github.com/dukkarya/ml-kasus-06-pm25-bandung-AbdanArya-integrasi--1-.git] |
| Video Demo | [Isi tautan video demonstrasi] |

## 1. Deskripsi Proyek

Proyek ini mengembangkan sistem Machine Learning untuk memprediksi konsentrasi partikulat halus PM2.5 tiga jam ke depan pada lima stasiun pemantauan kualitas udara di Bandung. Sistem dirancang untuk membantu masyarakat dan petugas dinas memperoleh gambaran mengenai kondisi kualitas udara sebelum waktu pengamatan berikutnya.

Model menggunakan pendekatan regresi deret waktu (*time-series regression*) dengan memanfaatkan data konsentrasi PM2.5, kondisi meteorologi, informasi waktu, dan riwayat pengamatan. Hasil prediksi diintegrasikan ke dalam aplikasi web berbasis Streamlit agar dapat digunakan melalui antarmuka yang sederhana.

## 2. Tujuan dan Ruang Lingkup

Tujuan pengembangan proyek ini adalah:

1. Membangun model untuk memprediksi konsentrasi PM2.5 pada tiga jam berikutnya.
2. Mengidentifikasi pola perubahan PM2.5 berdasarkan kondisi lingkungan dan waktu.
3. Membandingkan performa model Machine Learning dengan baseline persistence.
4. Mengevaluasi akurasi prediksi menggunakan Mean Absolute Error (MAE).
5. Mengukur kemampuan model dalam mendeteksi konsentrasi PM2.5 aktual di atas 55 µg/m³.
6. Mengintegrasikan model ke aplikasi web yang dapat digunakan untuk memperoleh hasil prakiraan.

**Lokasi pemantauan:**
- Dago
- Soekarno-Hatta
- Cibeureum
- Antapani
- Ujungberung

**Periode dataset:** Juli 2024–Juni 2026.

**Target prediksi:** nilai PM2.5 tepat tiga jam setelah waktu pengamatan pada stasiun yang sama.

## 3. Dataset dan Pengolahan Data

Dataset awal tersimpan dalam `data/raw.csv` dengan 87.650 baris pengamatan dan delapan kolom utama, yaitu waktu, stasiun, konsentrasi PM2.5, suhu, kelembapan, kecepatan angin, arah angin, dan curah hujan.

### Pengolahan data

| Permasalahan | Jumlah | Penanganan |
|---|---:|---|
| Variasi penulisan nama stasiun | 14 variasi | Standardisasi menjadi lima nama stasiun |
| Duplikasi identik | 398 baris | Dihapus |
| Konflik pasangan stasiun-waktu | 7 kunci atau 14 baris | Baris konflik dikeluarkan dari dataset pemodelan |
| Nilai sensor -999 | 173 nilai PM2.5 pada audit awal | Diubah menjadi missing value (NaN) |
| Baris setelah cleaning | 87.238 | Digunakan untuk pembentukan target |

Target PM2.5 tiga jam ke depan dibentuk menggunakan pencocokan timestamp secara tepat pada stasiun yang sama. Dari keseluruhan data hasil cleaning, terdapat 86.557 baris dengan target tersedia.

Nilai kosong pada fitur tidak diimputasi secara global sebelum pemodelan. Penanganan missing value dilakukan melalui pipeline preprocessing agar parameter imputasi hanya dipelajari dari data training.

### Pembagian dataset

Pembagian data dilakukan berdasarkan `target_time`, bukan secara acak.

| Dataset | Periode target | Jumlah target tersedia |
|---|---|---:|
| Train | Juli 2024–Februari 2026 | 72.043 |
| Validation | Maret–April 2026 | 7.262 |
| Test | Mei–Juni 2026 | 7.252 |

Test set disimpan untuk evaluasi akhir setelah pemilihan model dilakukan menggunakan validation.

## 4. Metodologi dan Pengembangan Model

Pengembangan model terdiri dari audit data, cleaning, pembentukan target, rekayasa fitur, pembagian dataset berdasarkan waktu, pembuatan baseline, pelatihan model, dan evaluasi.

### Feature Engineering

Fitur yang digunakan meliputi:

- **Fitur pengamatan saat ini:** PM2.5, suhu, kelembapan, kecepatan angin, dan curah hujan.
- **Fitur arah angin:** transformasi sinus dan cosinus untuk merepresentasikan arah secara siklik.
- **Fitur waktu:** jam, hari dalam minggu, dan bulan dalam bentuk siklik.
- **Fitur historis:** PM2.5 pada satu, tiga, dan 24 jam sebelumnya.
- **Fitur kategorikal:** nama stasiun pemantauan.

Fitur lag diperoleh melalui pencocokan stasiun dan timestamp historis yang tepat, sehingga tidak menggunakan informasi masa depan.

### Baseline dan Model

Baseline yang digunakan adalah *persistence*, yaitu menganggap nilai PM2.5 tiga jam berikutnya sama dengan konsentrasi PM2.5 saat ini.

Tiga algoritma Machine Learning dibandingkan:

1. Ridge Regression.
2. Random Forest Regressor.
3. HistGradientBoosting Regressor.

Preprocessing dilakukan menggunakan pipeline scikit-learn. Fitur numerik ditangani dengan imputasi median, sedangkan fitur kategorikal stasiun menggunakan imputasi modus dan One-Hot Encoding. Seluruh proses fitting preprocessing dilakukan menggunakan data train.

Pemilihan model dilakukan berdasarkan MAE validation terendah, dengan tetap memperhatikan performa setiap stasiun dan kemampuan deteksi peringatan.

## 5. Hasil Evaluasi Model

### Perbandingan pada validation

Evaluasi validation menggunakan 7.246 sampel yang sama untuk seluruh model dan baseline.

| Model | MAE (µg/m³) |
|---|---:|
| Persistence | 6,2476 |
| Ridge Regression | 4,9982 |
| Random Forest | 4,7236 |
| **HistGradientBoosting** | **4,6650** |

HistGradientBoosting memperoleh MAE terendah dan menunjukkan penurunan kesalahan sebesar 25,33% dibandingkan baseline persistence.

Model ini juga menghasilkan MAE lebih rendah daripada persistence pada seluruh lima stasiun.

### Evaluasi final pada test

Setelah model dipilih menggunakan validation, HistGradientBoosting digunakan untuk evaluasi final pada data test.

| Indikator | Hasil |
|---|---:|
| Model final | HistGradientBoosting |
| Jumlah sampel evaluasi bersama | 7.236 |
| MAE test model | 4,6348 µg/m³ |
| Peningkatan relatif terhadap baseline test | 29,33% |

Hasil tersebut menunjukkan bahwa model final memberikan kesalahan prediksi yang lebih rendah dibandingkan metode persistence pada subset evaluasi test yang sama.

### Evaluasi peringatan PM2.5

Selain MAE, model dievaluasi berdasarkan kemampuannya mendeteksi kejadian ketika konsentrasi PM2.5 aktual melebihi 55 µg/m³.

Pada validation, HistGradientBoosting memiliki recall peringatan sebesar 9,30%, precision sebesar 80%, dan menghasilkan satu false alarm. Hal ini menunjukkan bahwa prediksi model cukup selektif dalam memberikan alarm, tetapi masih terdapat banyak kejadian PM2.5 tinggi yang belum terdeteksi.

Karena itu, MAE yang lebih rendah belum berarti model sepenuhnya andal untuk kebutuhan peringatan dini. Kemampuan deteksi konsentrasi tinggi masih perlu ditingkatkan.

## 6. Aplikasi dan Integrasi Sistem

Model yang telah dilatih diintegrasikan ke dalam aplikasi web menggunakan Streamlit. Pengguna dapat memasukkan informasi stasiun, waktu pengamatan, konsentrasi PM2.5, serta kondisi meteorologi untuk memperoleh estimasi PM2.5 tiga jam berikutnya.

**Akses aplikasi:**  
https://ml-kasus-06-pm-uzs2bl.streamlit.app/

### Arsitektur sistem

```text
Dataset Pengamatan
        |
        v
Data Cleaning
        |
        v
Feature Engineering
        |
        v
Model Training & Evaluation
        |
        v
Model Artifact
(model.joblib + metadata.json)
        |
        v
Predictor
        |
        v
Streamlit Application
        |
        v
Prediksi PM2.5 +3 Jam
```

### Integrasi fitur historis

Aplikasi menggunakan `src/features.py` untuk membentuk fitur waktu dan arah angin. Sementara itu, `app/predictor.py` bertanggung jawab memuat model dan menghasilkan prediksi.

Untuk memperoleh fitur lag, aplikasi melakukan pencarian nilai PM2.5 historis dari `data/processed/pm25_bandung_clean.csv` berdasarkan stasiun yang sama pada waktu satu, tiga, dan 24 jam sebelumnya. Apabila data historis tidak ditemukan, nilai kosong ditangani oleh preprocessing dalam pipeline model.

### Menjalankan aplikasi secara lokal

Pastikan Python dan Git telah terpasang pada perangkat.

**1. Clone repository**

```bash
git clone https://github.com/USERNAME_GITHUB/ml-kasus-06-pm25-bandung-AbdanArya.git
```

Ganti `USERNAME_GITHUB` dengan username pemilik repository.

**2. Masuk ke folder proyek**

```bash
cd ml-kasus-06-pm25-bandung-AbdanArya
```

**3. Membuat virtual environment**

```bash
python -m venv .venv
```

Aktifkan pada Windows PowerShell:

```powershell
.venv\Scripts\activate
```

**4. Instalasi library**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**5. Menjalankan aplikasi**

```bash
streamlit run app/streamlit_app.py
```

Aplikasi dapat diakses melalui:

```text
http://localhost:8501
```

## 7. Struktur Repository

```text
ml-kasus-06-pm25-bandung-AbdanArya/
│
├── app/
│   ├── streamlit_app.py
│   └── predictor.py
│
├── artifacts/
│   ├── model.joblib
│   └── metadata.json
│
├── data/
│   ├── raw.csv
│   └── processed/
│       └── pm25_bandung_clean.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   └── features.py
│
├── tests/
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
└── README.md
```

## 8. Pembagian Tugas Anggota

Proyek dikerjakan oleh dua anggota melalui pembagian tanggung jawab yang saling melengkapi.

| Anggota | Nama | NIM | Tanggung jawab |
|---|---|---|---|
| Anggota 1 | | | Data preparation, EDA, cleaning, feature engineering, baseline, training, evaluasi, dan artifact model |
| Anggota 2 | | | Pengembangan Streamlit, validasi input, integrasi predictor, pengujian aplikasi, deployment, dan dokumentasi |

Kedua anggota bekerja sama dalam proses integrasi, pengujian sistem, penyusunan laporan, dan demonstrasi proyek.

## 9. Keterbatasan dan Pengembangan Selanjutnya

Aplikasi saat ini menggunakan arsip data historis hingga 30 Juni 2026 dan belum terhubung dengan sensor maupun API pengukuran kualitas udara secara real-time. Dengan demikian, aplikasi berfungsi sebagai demonstrasi atau simulasi prediksi berdasarkan data historis, bukan layanan prakiraan real-time.

Fitur lag juga bergantung pada ketersediaan data historis pada stasiun dan timestamp yang sesuai. Selain itu, kemampuan model dalam mendeteksi kejadian PM2.5 di atas 55 µg/m³ masih perlu ditingkatkan.

Pengembangan selanjutnya dapat mencakup integrasi sumber data real-time, penambahan data pengamatan, evaluasi berkala, peningkatan recall peringatan, serta pemantauan perubahan pola data agar kinerja model tetap terjaga.

## 10. Luaran Proyek

Luaran yang dihasilkan dari proyek ini meliputi:

1. Repository GitHub yang berisi source code, notebook, dataset pendukung, model, metadata, dan dokumentasi.
2. Model Machine Learning HistGradientBoosting untuk prakiraan PM2.5 tiga jam ke depan.
3. Aplikasi web berbasis Streamlit yang telah terintegrasi dengan model.
4. Laporan proyek Machine Learning.
5. Video demonstrasi penggunaan aplikasi.

## Referensi dan Tautan

- **Aplikasi:** https://ml-kasus-06-pm-uzs2bl.streamlit.app/
- **Repository:** [Isi tautan repository GitHub]
- **Video demonstrasi:** [Isi tautan video]

---

**Machine Learning Project — MKB UPI**  
**Kasus 06: Prakiraan PM2.5 Bandung**  
*Dikembangkan oleh Tim Abdan & Arya.*
