# D.I.A.N.A (Dynamic Imaging-based Adaptive Neural Assistant)
Sistem Cerdas Adaptive Radiotherapy (ART) Kanker Serviks: Mengatasi Variasi Anatomi Inter-Fraksi Pasien berbasis Machine Learning

# Prediksi Kebutuhan Radioterapi Adaptif dari CT Scan Awal

## 1. Tujuan Proyek

Penelitian ini bertujuan untuk mengembangkan sebuah model *deep learning* yang mampu memprediksi apakah seorang pasien kanker panggul akan memerlukan adaptasi rencana terapi radiasi di masa depan, dengan **hanya menganalisis citra CT (Computed Tomography) Perencanaan Awal**. Keberhasilan model ini berpotensi menjadi alat bantu klinis untuk mengidentifikasi pasien berisiko tinggi sejak dini, memungkinkan pemantauan yang lebih ketat atau penyesuaian strategi terapi.

---

## 2. Hasil Akhir

Setelah melalui serangkaian proses pra-pemrosesan, pelatihan, dan validasi yang ketat, model **Convolutional Neural Network (CNN) 3D** murni yang dikembangkan berhasil mencapai performa yang sangat menjanjikan pada set data uji yang terpisah.

| **Metrik Kunci** | **Skor pada Data Uji** |
| ----------------------- | ---------------------- |
| **Akurasi** | **88.00%** |
| **Recall (Sensitivitas)** | **100%** |
| **Presisi** | 73%                    |
| **F1-Score** | 84%                    |

**Kesimpulan Utama:** Model berhasil **mengidentifikasi dengan benar semua pasien (100%)** yang sebenarnya memerlukan adaptasi terapi, yang merupakan tujuan klinis terpenting dari penelitian ini.

---

## 3. Struktur Proyek

Proyek ini terdiri dari beberapa skrip Python yang harus dijalankan secara berurutan untuk mereplikasi hasil.


├── 01_konversi_final.py      # (Lokal) Mengonversi CT DICOM mentah ke format NIfTI.

├── 02_buat_label.py          # (Lokal) Membuat file label.txt otomatis untuk setiap pasien.

├── 03_bagi_dataset.py        # (Lokal) Membagi dataset menjadi set Latih, Validasi, & Uji.

├── komponen_cnn_final.py     # Modul berisi arsitektur model dan data loader.

├── latih_cnn_final.py        # (Colab) Skrip utama untuk melatih model menggunakan GPU.

├── uji_model_final.py        # (Colab) Skrip untuk menguji model terbaik pada data uji.

├── requirements.txt          # Daftar pustaka Python yang dibutuhkan.

└── README.md                 # Penjelasan proyek ini.

---

## 4. Alur Kerja & Cara Menjalankan

### Tahap 0: Prasyarat

1.  **Struktur Folder Awal:**
    * Buat folder utama proyek di komputer Anda, misal `D:/Proyek_AI/`.
    * Di dalamnya, buat folder `DATASET_MENTAH/` dan letakkan semua folder data DICOM pasien di sana (misal, `PA0`, `PA1`, dst.).
    * Letakkan file `peta_data.csv` (berisi `ID_Pasien` & `Grup`) di dalam `D:/Proyek_AI/`.

2.  **Instal Pustaka yang Dibutuhkan:**
    Buka terminal atau command prompt, arahkan ke folder proyek Anda, dan jalankan:
    ```bash
    pip install -r requirements.txt
    ```

### Tahap 1: Persiapan Data (Jalankan di Komputer Lokal/VS Code)

Jalankan skrip-skrip berikut secara berurutan. **Penting:** Pastikan untuk menyesuaikan variabel `PATH` di bagian atas setiap skrip agar sesuai dengan struktur folder Anda.

1.  **Konversi DICOM ke NIfTI:**
    ```bash
    python 01_konversi_final.py
    ```
    * **Output:** Akan membuat folder `DATASET_OLAHAN/` (atau nama lain yang Anda tentukan) berisi file NIfTI untuk setiap pasien.

2.  **Buat File Label:**
    ```bash
    python 02_buat_label.py
    ```
    * **Output:** Akan membuat file `label.txt` di dalam setiap subfolder pasien di `DATASET_OLAHAN/`.

3.  **Bagi Dataset:**
    ```bash
    python 03_bagi_dataset.py
    ```
    * **Output:** Akan membuat file `peta_data_final.csv` di folder utama Anda.

4.  **Sinkronkan ke Google Drive:**
    * Unggah seluruh folder `DATASET_OLAHAN/` yang berisi file NIfTI dan `label.txt`.
    * Unggah file `peta_data_final.csv`.
    * Unggah file `komponen_cnn_final.py`.

### Tahap 2: Pelatihan & Pengujian (Jalankan di Google Colab)

1.  **Siapkan Notebook Colab:**
    * Buat notebook baru di Google Colab.
    * Pastikan Anda memilih *runtime* dengan akselerator **GPU** (Runtime -> Change runtime type -> GPU).

2.  **Latih Model:**
    * Salin-tempel seluruh isi dari skrip `latih_cnn_final.py` ke dalam sel notebook Colab.
    * Pastikan variabel `DRIVE_BASE_PATH` dan path lainnya di dalam skrip sudah sesuai dengan lokasi data Anda di Google Drive.
    * Jalankan sel tersebut. Proses ini akan menghubungkan Drive, melatih model, dan menyimpan file `model_terbaik.h5` ke Google Drive Anda.

3.  **Uji Model:**
    * Setelah pelatihan selesai, salin-tempel seluruh isi dari skrip `uji_model_final.py` ke sel notebook baru.
    * Pastikan path-nya sudah benar.
    * Jalankan sel tersebut untuk mendapatkan laporan evaluasi akhir dan *confusion matrix* pada data uji.
