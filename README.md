# Lip Reading Experiments: Image-Based vs Landmark-Based

Repositori ini berisi dua *notebook* eksperimen untuk membangun model *Lip Reading* (Membaca Gerak Bibir). Eksperimen ini dirancang untuk dijalankan di **Google Colab** menggunakan dataset publik IndoLR.

Secara garis besar, kedua *notebook* memiliki kesamaan pada:
- **Titik Landmark:** Menggunakan **40 titik spesifik** (`LIPS_PAIRS`) dari MediaPipe yang merepresentasikan cincin bibir luar dan dalam.
- **Validasi:** Menerapkan skema **LOSO (Leave-One-Subject-Out) 8-Fold Cross-Validation** dengan rasio pembagian dataset: **6 Train / 1 Val / 1 Test**.

---

## Detail

### Notebook `LipReading_V5`: Pendekatan Berbasis Citra (Spatio-Temporal)
Versi ini menggunakan potongan gambar (ROI) bibir secara langsung sebagai input model dan menyimpan hasil *precompute* ke dalam format `.npy`.

- **Precompute & Crop:** 
  Mengekstrak *Bounding Box* persis mengikuti bentuk bibir (berdasarkan koordinat min/max landmark) dengan tambahan *padding* proporsional agar bibir tidak terpotong. Sistem ini tidak akan ikut menangkap hidung dagu.
- **Penyesuaian Frame:** 
  Disesuaikan untuk target 30 FPS.
  - *Jika frame berlebih:* Dilakukan *center crop* simetris (membuang frame awal & akhir yang cenderung diam).
  - *Jika frame kurang:* Frame asli dipertahankan di awal, dan *black-padding* ditambahkan **hanya di akhir sequence**.
- **Arsitektur Model:** 
  Menggunakan **LRCN (Long Recurrent Convolutional Network)** dengan optimasi layer *TimeDistributed* dan LSTM. Total parameter: **~1.263.000**.

### Notebook `LipReading_V6`: Pendekatan Berbasis Koordinat (Landmark + MAR)
Versi ini tidak menggunakan data gambar, melainkan hanya berfokus pada pergerakan titik koordinat bibir sehingga jauh lebih ringan dan efisien.

- **Precompute & Normalisasi:** 
  Input berupa koordinat $(x,y)$ yang telah dinormalisasi (rotasi, skala berdasarkan jarak antar-mata, dan translasi centroid bibir) ditambah 1 fitur **MAR (Mouth Aspect Ratio)**.
- **Smart Sequence Processing:** 
  Tidak lagi menggunakan *crop/pad* statis. V6 mendeteksi **segmen aktif bicara** berdasarkan nilai *threshold* MAR (menyaring *noise* seperti tarikan napas/decak bibir). Setelah segmen bicara didapat, sequence di-**resample (interpolasi linear)** agar ukurannya pas dengan target.
- **Arsitektur Model:** 
  Menggunakan **BiGRU dengan Attention Pooling**. Terdapat *Conv1D frontend* untuk menangkap kecepatan gerak antar-frame. *Attention* digunakan agar model otomatis fokus pada frame penting (puncak artikulasi).

---

## 🚀 Cara Penggunaan

1. *notebook* (`LipReading_V5` atau `LipReading_V6`) di rancang untuk dijalankan melalui **Google Colab**. sehingga menjalankannya di local perlu ada sedikit perbaikan dari cell config untuk mengubah path dan juga tidak melakukan atau menajalankan file backup yang memang diperuntukan untuk google colab karena butuh untuk mengganti runtime dari CPU -> GPU.
2. **Dataset (IndoLR) bersifat publik**. Anda tidak memerlukan API Key atau konfigurasi rahasia Kaggle. Dataset akan otomatis terunduh menggunakan `kagglehub` melalui *snippet* kode berikut yang sudah ada di dalam *notebook*:

```python
import kagglehub

kaggle_path = kagglehub.dataset_download("abasset/indolr")
print("Dataset terunduh di:", kaggle_path)
