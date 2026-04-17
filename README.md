# Klasifikasi Tingkat Tutur (Unggah-Ungguh) Bahasa Jawa

Repositori ini berisi eksperimen *Natural Language Processing* (NLP) untuk mengklasifikasikan tingkat tutur (unggah-ungguh) dalam Bahasa Jawa ke dalam empat kategori utama: **Ngoko, Ngoko Alus, Krama, dan Krama Alus**.

Proyek ini membandingkan tiga pendekatan dari era yang berbeda untuk melihat sejauh mana model *machine learning* dapat memahami hierarki sosial dan kerumitan semantik dalam bahasa daerah.

## 🚀 Model yang Dibandingkan

1. **TF-IDF + SVM:** Pendekatan statistik klasik berbasis frekuensi kata.
2. **FastText + BiLSTM:** Pendekatan *deep learning* menggunakan *word embedding* (berbasis subword/karakter) dan *sequence modeling* dengan mekanisme *Self-Attention*.
3. **Javanese BERT (`w11wo/javanese-roberta-small`):** Pendekatan *state-of-the-art* menggunakan *contextual embedding* dari Transformer yang telah di-*pretrain* khusus pada teks Bahasa Jawa.

## 📊 Dataset

Dataset yang digunakan diambil dari Hugging Face: [JavaneseHonorifics/Unggah-Ungguh](https://huggingface.co/datasets/JavaneseHonorifics/Unggah-Ungguh) (menggunakan konfigurasi `translation`).

## 📈 Ringkasan Hasil

Secara keseluruhan, pemahaman konteks adalah kunci dalam bahasa Jawa:
* **Javanese BERT** mencatatkan performa terbaik. Kemampuannya memahami konteks kalimat penuh membuatnya sangat unggul dalam membedakan register bahasa yang ambigu (seperti Krama Alus).
* **FastText + BiLSTM** memberikan hasil yang solid dengan kemampuan menangkap kemiripan makna dan menangani kosakata baru (OOV).
* **TF-IDF + SVM** menjadi *baseline* yang ringan dan tangguh dengan akurasi ~89%, meskipun memiliki kelemahan mendasar karena tidak dapat menangkap makna kontekstual antar kata.

*Catatan: Visualisasi evaluasi model seperti Bar Chart dan Confusion Matrix dapat dilihat di dalam folder `image/`.*

## 📂 Struktur Repositori

* `notebook.ipynb` : *Source code* utama (Jupyter Notebook) yang berisi alur lengkap dari *preprocessing*, *training*, hingga *evaluation* ketiga model.
* `artikel.md` : Dokumentasi lengkap dan *storytelling* dari eksperimen ini (draf artikel Medium).
* `image/` : Direktori penyimpanan grafik perbandingan akurasi dan *confusion matrix*.

## 🛠️ Cara Menjalankan Eksperimen

1. *Clone* repositori ini ke komputer lokalmu:
   ```bash
   git clone [https://github.com/USERNAME_GITHUB/NAMA_REPO.git](https://github.com/USERNAME_GITHUB/NAMA_REPO.git)
   cd NAMA_REPO
