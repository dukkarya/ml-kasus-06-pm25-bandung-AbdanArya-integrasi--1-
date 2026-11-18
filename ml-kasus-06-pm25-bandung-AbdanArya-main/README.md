# Prakiraan PM2.5 Bandung

**Machine Learning Project — Kasus 06 | MKB UPI**

Proyek Machine Learning untuk memprediksi konsentrasi partikulat halus PM2.5 tiga jam ke depan pada lima stasiun pemantauan kualitas udara di Bandung. Sistem ini dikembangkan sebagai sarana prakiraan kualitas udara untuk membantu masyarakat dan petugas dinas melakukan tindakan antisipatif sebelum kondisi udara memburuk.

Prediksi dibangun menggunakan pendekatan regresi deret waktu (*time-series regression*) berdasarkan data pengamatan kualitas udara dan kondisi meteorologi. Hasil prediksi direncanakan ditampilkan melalui aplikasi web berbasis Streamlit.

---

## 1. Latar Belakang

PM2.5 merupakan partikulat halus berdiameter 2,5 mikrometer atau lebih kecil yang dapat masuk ke saluran pernapasan. Perubahan konsentrasi PM2.5 dapat memengaruhi kualitas udara dan meningkatkan risiko kesehatan, terutama bagi kelompok sensitif seperti penderita gangguan pernapasan.

Pemantauan kualitas udara saja belum cukup untuk memberikan peringatan dini apabila informasi baru tersedia setelah kondisi udara memburuk. Oleh karena itu, proyek ini mengembangkan model yang dapat memperkirakan konsentrasi PM2.5 tiga jam setelah waktu pengamatan.

Sebagai contoh, data pengamatan pukul 15.00 digunakan untuk memperkirakan konsentrasi PM2.5 pada pukul 18.00 di stasiun yang sama.

## 2. Tujuan Proyek

1. Mengembangkan model Machine Learning untuk memprediksi PM2.5 tiga jam ke depan.
2. Mengidentifikasi pola perubahan kualitas udara berdasarkan data pengamatan dan kondisi meteorologi.
3. Membandingkan performa model dengan metode baseline.
4. Mengevaluasi akurasi prediksi menggunakan Mean Absolute Error (MAE).
5. Mengukur kemampuan model dalam memberikan peringatan ketika konsentrasi PM2.5 aktual melebihi 55 µg/m³.
6. Mengembangkan aplikasi web yang menampilkan hasil prakiraan dan informasi kualitas udara secara mudah dipahami.

## 3. Ruang Lingkup Proyek

### Kasus
Prakiraan PM2.5 Bandung.

### Jenis Machine Learning
Supervised Learning — Time-Series Regression.

### Target Prediksi
Nilai konsentrasi PM2.5 tiga jam setelah waktu pengamatan pada stasiun yang sama.

### Lokasi Pemantauan
Lima stasiun kualitas udara:

- Dago
- Soekarno-Hatta
- Cibeureum
- Antapani
- Ujungberung

### Periode Dataset
Juli 2024 – Juni 2026.

### Pengguna Aplikasi
- Masyarakat sebagai pengguna informasi prakiraan kualitas udara.
- Petugas dinas sebagai pengguna informasi pemantauan dan peringatan dini.

## 4. Dataset

Dataset utama tersedia dalam file:

`data/raw.csv`

Setiap baris merepresentasikan satu stasiun pemantauan pada satu waktu pengamatan.

### Atribut Dataset

| Kolom | Keterangan | Satuan |
|---|---|---|
| `waktu` | Waktu pengamatan | WIB |
| `stasiun` | Nama stasiun pemantauan | - |
| `pm25_ugm3` | Konsentrasi PM2.5 | µg/m³ |
| `suhu_C` | Suhu udara | °C |
| `kelembaban_pct` | Kelembapan udara | % |
| `kecepatan_angin_ms` | Kecepatan angin | m/s |
| `arah_angin_deg` | Arah angin | Derajat |
| `curah_hujan_mm` | Curah hujan | mm |

### Data Preparation

Dataset belum dibersihkan dan perlu melalui beberapa tahapan pemeriksaan:

- Pemeriksaan tipe dan format data.
- Penanganan missing values.
- Pemeriksaan data duplikat.
- Pemeriksaan konflik data pada pasangan stasiun dan waktu.
- Standardisasi penulisan nama stasiun.
- Identifikasi nilai sensor yang tidak valid.
- Pemeriksaan urutan waktu dan kelengkapan pengamatan.

Nilai sensor yang tidak valid tidak boleh langsung dianggap sebagai pengukuran sebenarnya.

## 5. Metodologi Machine Learning

Pengembangan model mengikuti tahapan berikut:

1. **Exploratory Data Analysis (EDA)**  
   Memahami struktur dataset, distribusi PM2.5, pola temporal, dan hubungan antara parameter meteorologi.

2. **Data Cleaning**  
   Memperbaiki kualitas data, menangani nilai kosong, duplikasi, dan kesalahan pengukuran.

3. **Target Engineering**  
   Membentuk target PM2.5 tiga jam ke depan untuk setiap stasiun berdasarkan waktu pengamatan.

4. **Feature Engineering**  
   Mengembangkan fitur yang relevan dari kondisi pengamatan saat ini dan informasi historis yang tersedia.

5. **Data Splitting**  
   Membagi dataset berdasarkan waktu untuk mencegah kebocoran informasi masa depan.

6. **Baseline Modeling**  
   Menggunakan metode persistence, yaitu menganggap PM2.5 tiga jam berikutnya sama dengan nilai PM2.5 saat ini.

7. **Model Training**  
   Melatih dan membandingkan beberapa algoritma regresi.

8. **Model Evaluation**  
   Mengevaluasi performa model dan kemampuan menghasilkan peringatan dini.

### Pembagian Dataset

Pembagian data mengikuti ketentuan kasus:

| Dataset | Periode | Kegunaan |
|---|---|---|
| Train | Juli 2024 – Februari 2026 | Melatih model |
| Validation | Maret – April 2026 | Memilih model dan mengevaluasi pengaturan |
| Test | Mei – Juni 2026 | Evaluasi akhir model |

**Catatan:** Pembagian data wajib berdasarkan waktu, bukan random split. Test set hanya digunakan satu kali pada tahap evaluasi akhir.

### Baseline

Metode pembanding yang digunakan adalah persistence:

Prediksi PM2.5 tiga jam ke depan = PM2.5 saat ini.

### Metrik Evaluasi

**Mean Absolute Error (MAE)**

Digunakan sebagai metrik utama untuk mengukur rata-rata kesalahan prediksi dalam satuan µg/m³. Semakin kecil nilai MAE, semakin baik performa prediksi.

**Keberhasilan Peringatan PM2.5**

Model juga dievaluasi berdasarkan kemampuannya mendeteksi pengamatan dengan konsentrasi PM2.5 aktual di atas 55 µg/m³.

Salah satu ukuran yang digunakan adalah recall peringatan, yaitu proporsi kejadian aktual di atas ambang tersebut yang berhasil diprediksi sebagai kondisi di atas ambang.

## 6. Teknologi yang Digunakan

| Teknologi | Fungsi |
|---|---|
| Python | Bahasa pemrograman utama |
| Pandas | Pengolahan dataset |
| NumPy | Komputasi numerik |
| Matplotlib | Visualisasi data |
| Seaborn | Analisis visualisasi |
| Scikit-learn | Pengembangan model Machine Learning |
| Joblib | Penyimpanan model |
| Streamlit | Pengembangan aplikasi web |
| Git & GitHub | Version control dan kolaborasi |
| VS Code | Lingkungan pengembangan |

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
│   └── raw.csv
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
├── .gitignore
├── README.md
└── requirements.txt
```

Keterangan:

- `app/`: Berisi aplikasi Streamlit dan fungsi pemanggilan model.
- `artifacts/`: Menyimpan model terlatih beserta metadata.
- `data/`: Menyimpan dataset.
- `notebooks/`: Menyimpan notebook untuk EDA dan pemodelan.
- `src/`: Menyimpan fungsi feature engineering yang digunakan dalam training dan prediksi.
- `tests/`: Menyimpan pengujian model dan aplikasi.
- `.streamlit/`: Menyimpan konfigurasi tampilan Streamlit.

File model dan metadata akan dihasilkan setelah proses training selesai.

## 8. Instalasi dan Persiapan

### Prasyarat

Pastikan perangkat sudah memiliki:

- Python
- Git
- Visual Studio Code
- Koneksi internet untuk instalasi library

### Clone Repository

```bash
git clone https://github.com/USERNAME_GITHUB/ml-kasus-06-pm25-bandung-AbdanArya.git
```

Ganti `USERNAME_GITHUB` dengan username pemilik repository.

Masuk ke direktori proyek:

```bash
cd ml-kasus-06-pm25-bandung-AbdanArya
```

### Membuat Virtual Environment

```bash
python -m venv .venv
```

Aktifkan pada Windows PowerShell:

```powershell
.venv\Scripts\activate
```

### Instalasi Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 9. Menjalankan Aplikasi

Pastikan virtual environment sudah aktif dan seluruh dependencies berhasil diinstal.

Jalankan aplikasi dengan:

```bash
streamlit run app/streamlit_app.py
```

Aplikasi dapat dibuka melalui alamat lokal:

```text
http://localhost:8501
```

### Cara Menggunakan Aplikasi

1. Pilih stasiun pemantauan.
2. Masukkan tanggal dan waktu pengamatan.
3. Masukkan nilai PM2.5 dan parameter meteorologi.
4. Jalankan validasi data.
5. Setelah model terintegrasi, sistem akan menampilkan prakiraan PM2.5 tiga jam ke depan, kategori kualitas udara, dan saran tindakan.

## 10. Arsitektur Sistem

Alur pengembangan sistem direncanakan sebagai berikut:

```text
Data Pengamatan
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
Prakiraan PM2.5 +3 Jam
      |
      v
Informasi Kualitas Udara
dan Peringatan Dini
```

Feature engineering dan preprocessing pada aplikasi harus menggunakan proses yang konsisten dengan proses saat training agar tidak terjadi perbedaan antara data pelatihan dan data prediksi.

## 11. Pembagian Tugas Tim

Pengembangan proyek dilakukan oleh dua anggota dengan pendekatan kerja paralel.

### Anggota 1 — Data & Machine Learning

Bertanggung jawab atas:

- Exploratory Data Analysis.
- Data cleaning dan preprocessing.
- Target engineering.
- Feature engineering.
- Pembagian dataset berdasarkan waktu.
- Baseline modeling.
- Training dan pemilihan model.
- Evaluasi model.
- Ekspor `model.joblib` dan `metadata.json`.
- Dokumentasi hasil eksperimen.

### Anggota 2 — Application Development & Integration

Bertanggung jawab atas:

- Pengembangan antarmuka Streamlit.
- Validasi input pengguna.
- Persiapan predictor.
- Integrasi model ke aplikasi.
- Penyajian hasil prediksi dan rekomendasi.
- Pengujian aplikasi.
- Deployment.
- README dan dokumentasi proyek.
- Persiapan video demonstrasi.

Kedua anggota bertanggung jawab melakukan koordinasi dan pengujian integrasi secara bersama.

## 12. Status Pengembangan

Status proyek diperbarui sesuai perkembangan pengerjaan.

| Komponen | Status |
|---|---|
| Repository GitHub | Selesai |
| Struktur folder proyek | Selesai |
| Requirements dan environment | Selesai |
| Prototype aplikasi Streamlit | Selesai |
| Penyempurnaan UI/UX | Dalam pengembangan |
| EDA dan audit data | Dalam pengerjaan |
| Data cleaning | Menunggu hasil EDA |
| Feature engineering | Menunggu proses modeling |
| Baseline dan model training | Menunggu |
| Evaluasi model | Menunggu |
| Predictor dan integrasi model | Menunggu artifact |
| Deployment | Belum dilakukan |
| Laporan dan video demo | Belum diselesaikan |

## 13. Ketentuan dan Catatan Pengembangan

Beberapa prinsip yang harus diperhatikan selama pengerjaan:

- Gunakan informasi yang tersedia sampai waktu pengamatan saat melakukan prediksi.
- Jangan menggunakan informasi masa depan sebagai fitur.
- Target harus berasal dari pengamatan tiga jam setelah waktu saat ini pada stasiun yang sama.
- Gunakan pembagian dataset berdasarkan waktu.
- Bandingkan model dengan baseline persistence.
- Gunakan MAE sebagai metrik evaluasi utama.
- Jangan membuka test set berulang kali untuk memilih model.
- Pastikan preprocessing pada training dan prediksi konsisten.
- Jangan memasukkan file virtual environment `.venv` ke repository.
- Dokumentasikan versi library yang digunakan untuk training dan deployment.

## 14. Luaran Proyek

Luaran akhir yang ditargetkan meliputi:

1. Repository GitHub berisi notebook, source code, model artifact, metadata, requirements, dan dokumentasi.
2. Model Machine Learning yang dapat memprediksi PM2.5 tiga jam ke depan.
3. Aplikasi web yang dapat diakses melalui tautan.
4. Laporan proyek maksimal 6 halaman.
5. Video demonstrasi maksimal 5 menit.

---

**Proyek Machine Learning — MKB UPI**  
**Kasus 06: Prakiraan PM2.5 Bandung**

*Dikembangkan oleh Tim Abdan & Arya.*

---

## 15. Integrasi model final dan sumber fitur lag

Aplikasi menggunakan artifact final dari Tahap H:

- `artifacts/model.joblib`
- `artifacts/metadata.json`

Dataset historis yang dipakai untuk lookup fitur lag disimpan di:

`data/processed/pm25_bandung_clean.csv`

Untuk setiap prediksi pada waktu `t`, aplikasi mencari PM2.5 pada stasiun yang sama secara tepat pada `t-1 jam`, `t-3 jam`, dan `t-24 jam`. Pencarian menggunakan pasangan stasiun dan timestamp yang tepat, bukan baris terdekat atau pergeseran indeks. Jika nilai historis tidak tersedia, fitur diisi sebagai nilai kosong dan ditangani oleh imputasi di dalam pipeline model yang sudah dilatih.

Fitur kalender dan arah angin dibentuk oleh `src/features.py`, sedangkan pemuatan artifact dan prediksi dilakukan oleh `app/predictor.py`. Aplikasi tidak melakukan fitting ulang dan tidak membuat pipeline preprocessing terpisah.

### Menjalankan aplikasi

```bash
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Catatan penggunaan

Dataset yang disertakan merupakan arsip historis hingga 30 Juni 2026. Karena belum ada koneksi sensor/API real-time, aplikasi saat ini berfungsi sebagai demonstrasi/simulasi prediksi pada rentang waktu arsip tersebut. Pengguna memasukkan kondisi sensor pada waktu `t`; fitur lag diperoleh dari arsip historis stasiun yang sama. Jangan mengartikan hasil aplikasi sebagai prakiraan kualitas udara real-time.

Versi scikit-learn dan joblib dikunci mengikuti metadata artifact Tahap F untuk mengurangi risiko inkompatibilitas saat model dimuat.
