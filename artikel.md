# Ketika AI Belajar Berbahasa Jawa: Membandingkan TF-IDF, FastText+BiLSTM, dan Javanese BERT untuk Klasifikasi Unggah-Ungguh Bahasa Jawa

> *Sebuah perjalanan dari pendekatan klasik hingga transformer untuk memahami tingkatan bahasa paling kompleks di Nusantara*



---

Bahasa Jawa bukan sekadar bahasa biasa. Di dalamnya terkandung sistem **unggah-ungguh** — hierarki tutur yang mencerminkan kedudukan sosial, rasa hormat, dan kedalaman budaya Jawa yang telah diwariskan selama berabad-abad. Seorang penutur yang baik harus tahu kapan harus berkata *"mangan"* (Ngoko) dan kapan harus berkata *"dhahar"* (Krama Inggil) — dan kesalahan kecil bisa bermakna besar secara sosial.

Pertanyaannya: **bisakah mesin belajar memahami kerumitan ini?**

Di sinilah eksperimen ini dimulai. Kami membandingkan tiga pendekatan NLP yang mewakili tiga era berbeda dalam machine learning — dari metode klasik berbasis statistik hingga model transformer pra-latih khusus Bahasa Jawa — untuk menjawab satu tantangan: **mengklasifikasikan tingkat tutur Bahasa Jawa secara otomatis**.

---

## Memahami Masalahnya: Apa Itu Unggah-Ungguh?

Sebelum masuk ke kode dan angka, penting untuk memahami mengapa masalah ini menarik secara linguistik dan teknis sekaligus.

Unggah-ungguh Bahasa Jawa terbagi dalam empat level utama:

| Kode | Level | Konteks Penggunaan |
|------|-------|-------------------|
| **0** | **Ngoko** | Berbicara dengan teman sebaya, yang lebih muda, atau suasana santai |
| **1** | **Ngoko Alus** | Ngoko yang diperhalus, campuran dengan kosakata Krama |
| **2** | **Krama** | Formal, digunakan kepada orang yang lebih tua atau dihormati |
| **3** | **Krama Alus** | Tingkat paling tinggi dan paling formal |

Bagi mesin, membedakan empat level ini bukan perkara mudah. Kalimat *"Kowe arep mangan apa?"* (Ngoko) dan *"Panjenengan badhe dhahar menapa?"* (Krama Alus) mengandung makna yang sama persis — *"Kamu mau makan apa?"* — namun dengan pemilihan kata yang sama sekali berbeda. Inilah yang membuat klasifikasi unggah-ungguh menjadi benchmark yang ideal untuk menguji kemampuan model NLP dalam memahami semantik Bahasa Jawa.

---

## Dataset: Unggah-Ungguh dari Hugging Face

Dataset yang digunakan adalah **Unggah-Ungguh Bahasa Jawa** yang tersedia secara publik di Hugging Face dengan konfigurasi `translation`. Dataset ini memuat ribuan kalimat dalam Bahasa Jawa beserta terjemahannya dalam Bahasa Indonesia dan Inggris, dilengkapi label tingkat tutur.

```python
from datasets import load_dataset

dataset = load_dataset("JavaneseHonorifics/Unggah-Ungguh", "translation")
data = dataset["train"]
```

Struktur dataset setelah pembersihan:

| Kolom | Deskripsi |
|-------|-----------|
| `label` | Tingkat tutur (0–3) |
| `javanese` | Kalimat asli Bahasa Jawa |
| `javanese_clean` | Kalimat setelah preprocessing |
| `indonesian` | Terjemahan Bahasa Indonesia |
| `english` | Terjemahan Bahasa Inggris |

Kolom utama yang digunakan untuk pelatihan adalah `javanese_clean` dan `label`.

---

## Preprocessing: Membersihkan Teks Bahasa Jawa

Sebelum data dapat digunakan oleh model apapun, teks perlu dibersihkan. Tiga tahap preprocessing diterapkan:

1. **Lowercase** — Menyeragamkan kapitalisasi
2. **Hapus karakter non-alfabet** — Tanda baca, angka, dan simbol dihilangkan
3. **Normalisasi spasi** — Menghilangkan spasi ganda yang tidak perlu

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["javanese_clean"] = df["javanese"].apply(clean_text)
```

Dataset kemudian dibagi menjadi **80% data latih** dan **20% data uji** dengan stratifikasi kelas untuk menjaga distribusi yang proporsional.

---

## Tiga Pendekatan, Tiga Era NLP

Eksperimen ini dirancang untuk merepresentasikan evolusi alami dalam NLP:

```
Era 1: Statistik          Era 2: Word Embedding      Era 3: Transformer
TF-IDF + SVM      →      FastText + BiLSTM    →     Javanese BERT
(Tradisional)            (Deep Learning)             (Kontekstual)
```

Mari kita bedah satu per satu.

---

## Model 1: TF-IDF + SVM — Sang Veteran yang Tangguh

### Bagaimana Cara Kerjanya?

**TF-IDF** (*Term Frequency - Inverse Document Frequency*) adalah metode representasi teks yang sudah teruji selama dekade. Idenya sederhana namun cerdas: sebuah kata penting bukan hanya karena sering muncul di satu dokumen (*TF*), tapi juga karena jarang muncul di dokumen lain (*IDF*).

- **TF(t,d)** = frekuensi kata *t* dalam dokumen *d* ÷ total kata dalam *d*
- **IDF(t)** = log(N ÷ df(t)), di mana N = total dokumen, df(t) = dokumen yang mengandung *t*
- **TF-IDF(t,d)** = TF × IDF

Hasilnya adalah matriks vektor *sparse* berukuran besar (5000 fitur dalam eksperimen ini), di mana setiap kalimat direpresentasikan sebagai titik dalam ruang vektor berdasarkan kata-kata yang dikandungnya.

**SVM** (*Support Vector Machine*) kemudian mencari *hyperplane* optimal yang memisahkan keempat kelas tingkat tutur dalam ruang vektor tersebut.

```python
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
```

### Hasil Evaluasi

![Confusion Matrix TF-IDF + SVM](image/coffusion%20matrix%20tf%20idf.png)

![Bar Chart TF-IDF](image/bar%20chart%20tf%20idf.png)

TF-IDF + SVM mencatatkan akurasi sekitar **89%** — angka yang sangat respektabel untuk metode klasik. Analisis per kelas menunjukkan pola yang menarik:

- **Ngoko (F1 ~0.94)** — Performa terbaik. Kosakata Ngoko sangat khas dan mudah dikenali oleh model berbasis frekuensi kata.
- **Krama (F1 ~0.92)** — Kosakata formal Krama juga cukup konsisten sehingga mudah diidentifikasi.
- **Ngoko Alus (F1 ~0.83)** — Mulai ada kesalahan klasifikasi karena Ngoko Alus merupakan campuran dua register.
- **Krama Alus (F1 ~0.70)** — Performa paling lemah. Recall rendah menunjukkan banyak kasus Krama Alus tidak terdeteksi dengan benar.

**Kelemahan mendasar TF-IDF:** Model ini hanya menghitung *frekuensi kata*, tidak peduli *makna* atau *konteks*. Ketika dua kata memiliki makna serupa tapi tulisan berbeda (misalnya sinonim lintas register), TF-IDF tidak bisa menghubungkannya.

---

## Model 2: FastText + BiLSTM — Word Embedding Bertemu Deep Learning

### Langkah Maju: Dari Frekuensi ke Makna

**FastText** adalah word embedding yang dikembangkan Facebook AI Research. Berbeda dari TF-IDF yang melihat kata sebagai entitas tunggal, FastText memecah setiap kata menjadi *character n-gram* dan mempelajari representasi vektor yang menangkap kemiripan makna.

Contoh: kata `mangan` dipecah menjadi `<ma, man, ang, nga, gan, an>`. Ini berarti:
- Kata-kata yang morfologis mirip akan memiliki vektor berdekatan
- Kata **OOV** (*Out-Of-Vocabulary*) yang tidak ada di kamus tetap bisa direpresentasikan melalui karakter pembentuknya

| Aspek | TF-IDF | FastText |
|-------|--------|----------|
| Representasi | Sparse (frekuensi) | Dense (makna) |
| Hubungan kata | Tidak menangkap | Menangkap kemiripan semantik |
| Kata baru (OOV) | Tidak bisa | Bisa (via subword) |
| Dimensi | ~5000 (sparse) | 100 (dense) |

```python
ft_model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,           # Skip-gram
    epochs=30,
    min_n=2, max_n=5,  # Character n-gram range
    seed=42
)
```

### BiLSTM + Attention: Memahami Urutan Kata

Setelah setiap kalimat dikonversi menjadi vektor rata-rata (*mean pooling*) dari semua vektor kata, **BiLSTM** (*Bidirectional Long Short-Term Memory*) memproses representasi ini dari dua arah — maju dan mundur — untuk menangkap konteks yang lebih kaya.

Mekanisme **self-attention** kemudian memberi bobot lebih pada bagian teks yang paling informatif untuk tugas klasifikasi.

Arsitektur model:

```
Input (100-dim FastText vector)
    ↓
Batch Normalization
    ↓
Stacked BiLSTM (2 layer, 128 hidden units, bidirectional)
    ↓
Self-Attention (memilih bagian teks paling relevan)
    ↓
Classifier Head (Linear → LayerNorm → ReLU → Dropout → Linear)
    ↓
Output (4 kelas)
```

Training dikonfigurasi dengan beberapa teknik modern:
- **Loss**: CrossEntropyLoss dengan *class weight* + *label smoothing* 0.1
- **Optimizer**: AdamW dengan weight decay
- **Scheduler**: Cosine Annealing LR
- **Early Stopping**: patience = 10 epoch

### Hasil Evaluasi

![Confusion Matrix FastText + BiLSTM](image/coffusion%20matrix%20fast%20text%20bilstm.png)

![Bar Chart FastText BiLSTM](image/bar%20chart%20fast%20text%20bilstm.png)

FastText + BiLSTM menunjukkan peningkatan, khususnya pada kelas-kelas yang sulit dibedakan secara leksikal. Kemampuan memahami kemiripan antar kata menjadikannya lebih baik dalam menangani variasi kosakata antar register bahasa Jawa.

---

## Model 3: Javanese BERT — Kekuatan Contextual Embedding

### Revolusi Transformer: Satu Kata, Banyak Makna

Ini adalah puncak evolusi dalam eksperimen ini. **BERT** (*Bidirectional Encoder Representations from Transformers*) mengubah paradigma representasi kata secara fundamental.

Dalam FastText, kata *"gede"* selalu memiliki vektor yang sama, tidak peduli konteksnya. Dalam BERT, representasi kata **berubah sesuai konteks kalimat**. Kata yang sama bisa memiliki vektor berbeda tergantung kata-kata di sekitarnya.

Model yang dipilih adalah **`w11wo/javanese-roberta-small`** — varian RoBERTa yang telah di-*pretrain* secara khusus pada korpus Bahasa Jawa. Ini adalah keputusan krusial: menggunakan model yang sudah memahami distribusi linguistik Bahasa Jawa secara mendalam.

### Arsitektur Fine-Tuning

Strategi fine-tuning menggunakan **differential learning rate** — BERT backbone dilatih dengan learning rate kecil (2e-5) agar tidak "lupa" pengetahuan pretraining-nya, sementara classifier head dilatih lebih agresif (1e-4):

```python
optimizer_bert = torch.optim.AdamW([
    {'params': list(bert_model.bert.parameters()),       'lr': 2e-5, 'weight_decay': 0.01},
    {'params': list(bert_model.classifier.parameters()), 'lr': 1e-4, 'weight_decay': 0.01}
])
```

Arsitektur classifier head:

```
[CLS] token dari BERT (representasi seluruh kalimat)
    ↓
Linear (hidden → hidden/2)
    ↓
LayerNorm → GELU → Dropout(0.3)
    ↓
Linear (hidden/2 → 4 kelas)
```

**Linear warmup scheduler** digunakan pada 10% pertama training untuk menghindari perubahan bobot yang terlalu drastis di awal fine-tuning.

### Hasil Evaluasi

![Confusion Matrix Javanese BERT](image/coffusion%20matrix%20javanese%20bert.png)

![Bar Chart Javanese BERT](image/bar%20chat%20javanese%20bert.png)

Javanese BERT tampil sebagai model terkuat, terutama pada kelas-kelas yang secara linguistik paling ambigu — Ngoko Alus dan Krama Alus. Kemampuan memahami konteks penuh kalimat memungkinkan model ini membedakan nuansa halus yang tidak bisa ditangkap oleh pendekatan berbasis frekuensi atau bahkan word embedding statis.

---

## Perbandingan Final: Siapa yang Paling Mengerti Unggah-Ungguh?

![Perbandingan Akurasi 3 Model](image/perbandingan%20%20akurasi%203%20model.png)

| Model | Pendekatan | Akurasi | Kelebihan | Kekurangan |
|-------|-----------|---------|-----------|------------|
| **TF-IDF + SVM** | Klasik / Statistik | ~89% | Cepat, ringan, mudah diinterpretasi | Tidak menangkap makna, gagal di kelas ambigu |
| **FastText + BiLSTM** | Word Embedding + DL | >89% | Semantik kata, handle OOV, bidirectional | Representasi statis, tidak kontekstual |
| **Javanese BERT** | Contextual Transformer | Tertinggi | Kontekstual penuh, pretrained Bahasa Jawa | Komputasi berat, butuh GPU |

Javanese BERT secara konsisten unggul di semua kelas, terutama pada kelas yang paling sulit: **Krama Alus**. Ini masuk akal karena Krama Alus menggunakan kosakata yang sangat spesifik dan penempatannya sangat bergantung pada konteks kalimat secara keseluruhan.

---

## Apa yang Bisa Kita Pelajari?

Eksperimen ini bukan sekadar kompetisi angka akurasi. Ada beberapa insight penting yang bisa diambil:

**1. Kompleksitas linguistik membutuhkan representasi yang sepadan.**
TF-IDF yang hanya mengandalkan frekuensi kata sudah cukup baik untuk kelas-kelas yang punya kosakata khas (Ngoko, Krama). Tapi untuk level menengah seperti Ngoko Alus yang merupakan *campuran* register, model butuh pemahaman semantik yang lebih dalam.

**2. Pretrained model untuk bahasa lokal adalah game-changer.**
Keputusan menggunakan `w11wo/javanese-roberta-small` alih-alih BERT generik terbukti strategis. Model yang sudah "paham" distribusi linguistik Bahasa Jawa memberikan starting point yang jauh lebih baik untuk fine-tuning.

**3. Trade-off selalu ada.**
Javanese BERT memang terbaik secara akurasi, tapi butuh GPU dan waktu inferensi yang lebih lama. Untuk aplikasi real-time atau perangkat terbatas, TF-IDF + SVM yang menghasilkan 89% akurasi dengan biaya komputasi minimal mungkin adalah pilihan yang lebih pragmatis.

**4. Data berkualitas adalah fondasi segalanya.**
Kualitas klasifikasi yang tinggi juga bergantung pada kualitas dataset. Dataset Unggah-Ungguh yang terstruktur dengan baik dan berlabel konsisten memungkinkan ketiga model untuk belajar dengan optimal.

---

## Kesimpulan

Perjalanan dari TF-IDF menuju Javanese BERT adalah cerminan dari evolusi NLP itu sendiri — dari representasi statistik ke pemahaman semantik, lalu ke pemahaman kontekstual yang sesungguhnya.

Untuk tugas klasifikasi unggah-ungguh Bahasa Jawa, **Javanese BERT tampil sebagai pemenang yang jelas**, terutama karena kemampuannya memahami konteks penuh dan memanfaatkan pengetahuan linguistik Bahasa Jawa yang sudah terintegrasi dalam proses pretraining.

Namun yang lebih menarik dari angka akurasi adalah implikasinya: kita kini selangkah lebih dekat pada sistem NLP yang benar-benar bisa memahami kekayaan dan kerumitan bahasa-bahasa lokal Indonesia. Unggah-ungguh Bahasa Jawa hanyalah satu contoh — ada ratusan bahasa daerah dengan kompleksitasnya masing-masing yang menunggu untuk dijamah oleh teknologi.

Pertanyaannya bukan lagi *bisakah mesin memahami bahasa Jawa* — melainkan *seberapa dalam mesin bisa memahaminya*.

---

## Referensi & Resources

- Dataset: [JavaneseHonorifics/Unggah-Ungguh — Hugging Face](https://huggingface.co/datasets/JavaneseHonorifics/Unggah-Ungguh)
- Pretrained Model: [w11wo/javanese-roberta-small — Hugging Face](https://huggingface.co/w11wo/javanese-roberta-small)
- Libraries: `scikit-learn`, `gensim`, `PyTorch`, `Transformers (HuggingFace)`

---

*Notebook lengkap tersedia di repositori GitHub. Seluruh eksperimen dijalankan menggunakan kaggle dengan GPU T4.*

---

*Apakah artikel ini bermanfaat? Tinggalkan komentar atau clap jika kamu tertarik melihat eksperimen serupa untuk bahasa daerah lainnya!* 
