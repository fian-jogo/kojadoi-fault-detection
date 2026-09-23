# 05_preprocessing_and_lstm.md

## Preprocessing Data dan Baseline LSTM untuk Deteksi Fault PLTS

### 1. Tujuan

Blok ini bertujuan membangun **pipeline end-to-end pertama** yang dapat menerima dataset fault sintetik dari Hari 2 dan menghasilkan prediksi deteksi fault beserta lokalisasi node yang terdampak. Fokus utama Hari 3 bukan mencapai performa optimal, melainkan memvalidasi bahwa seluruh tahapan — dari loading data hingga evaluasi metrik — berjalan dengan benar dan dapat direproduksi.

Baseline LSTM ini akan menjadi **titik referensi** untuk mengukur peningkatan yang diberikan oleh komponen Graph Attention Network (GAT) dan Physics-Informed Neural Network (PINN) di Hari 4. Tanpa baseline yang jelas, klaim peningkatan performa dari arsitektur yang lebih kompleks tidak dapat dipertanggungjawabkan secara ilmiah.

---

### 2. Preprocessing Data

#### 2.1 Konfigurasi Tensor

Dataset `fault_dataset.csv` (216.000 baris, format long) dipivot menjadi tensor empat dimensi:

\[
\mathbf{X} \in \mathbb{R}^{N_{scenario} \times T \times N_{node} \times F}
\]

dengan:
- \( N_{scenario} = 180 \) — jumlah skenario
- \( T = 100 \) — panjang time series per skenario
- \( N_{node} = 12 \) — jumlah node (3 string × 4 panel)
- \( F = 5 \) — fitur per node (V, I, P, G, T)

Hasil pivot: `X.shape = (180, 100, 12, 5)`. Total elemen: 10,8 juta nilai float32.

#### 2.2 Normalisasi

Normalisasi Z-score diterapkan per fitur, dengan **statistik hanya dihitung dari training set** untuk menghindari data leakage:

\[
\tilde{X}_f = \frac{X_f - \mu_f}{\sigma_f}
\]

Statistik yang diperoleh dari training set:

| Fitur | Mean (\( \mu \)) | Std (\( \sigma \)) |
|---|---|---|
| V (V) | 39,055 | 2,382 |
| I (A) | 4,709 | 2,759 |
| P (W) | 186,305 | 110,082 |
| G (W/m²) | 463,949 | 247,227 |
| T (°C) | 28,318 | 0,708 |

Setelah normalisasi, distribusi `X_train` memiliki mean = 0,005 dan std = 1,031 — mendekati distribusi normal standar. Nilai minimum -15,658 dan maksimum 4,677 adalah outlier dari kondisi iradiansi rendah dan fault magnitude tinggi, yang tetap dipertahankan karena mengandung informasi penting untuk deteksi.

#### 2.3 Sliding Window

Time series 100 time step dipecah menjadi window dengan panjang \( T_{window} = 30 \) dan stride 1. Setiap window menghasilkan satu sample dengan label dari **time step terakhir** window.

Jumlah window per skenario: \( (100 - 30) / 1 + 1 = 71 \).

Total window: \( 180 \times 71 = 12.780 \). Distribusi per split:

| Split | Jumlah Window | Window Fault Aktif | Persentase |
|---|---|---|---|
| Train | 8.946 | 2.695 | 30,1% |
| Validation | 1.775 | 499 | 28,1% |
| Test | 2.059 | 617 | 30,0% |

Distribusi kelas yang seimbang antar split (~30% positif) mengonfirmasi bahwa stratifikasi bekerja dengan baik.

#### 2.4 Adjacency Matrix

Meskipun baseline LSTM belum menggunakan informasi graf, adjacency matrix dibangun sekarang untuk dipakai di Hari 4:

\[
A_{ij} = \begin{cases}
1 & \text{jika node } i \text{ dan } j \text{ terhubung} \\
0 & \text{lainnya}
\end{cases}
\]

Konektivitas yang dimodelkan:
- **Series edge**: dalam satu string, node \( n \) terhubung ke node \( n+1 \)
- **Cross-string edge**: node pada posisi yang sama di string berbeda saling terhubung (merepresentasikan kopling melalui bus DC)
- **Self-loop**: setiap node terhubung ke dirinya sendiri

Normalisasi simetris diterapkan: \( \hat{A} = D^{-1/2} A D^{-1/2} \), di mana \( D \) adalah matriks derajat.

---

### 3. Pembagian Dataset

Split dilakukan **by scenario**, bukan by window, untuk mencegah data leakage. Semua window dari satu skenario masuk ke split yang sama. Stratifikasi berdasarkan jenis fault:

| Split | Normal | LLF | LGF | PSC | Total |
|---|---|---|---|---|---|
| Train | 21 | 35 | 35 | 35 | 126 |
| Validation | 4 | 7 | 7 | 7 | 25 |
| Test | 5 | 8 | 8 | 8 | 29 |
| **Total** | **30** | **50** | **50** | **50** | **180** |

Verifikasi overlap: **0 skenario** yang muncul di lebih dari satu split, dikonfirmasi untuk semua pasangan (train-val, train-test, val-test).

---

### 4. Arsitektur Model Baseline

#### 4.1 LSTM Detector

Model baseline terdiri dari:

1. **Shared LSTM encoder**: LSTM 2-layer dengan hidden size 64, dropout 0.2. Input setiap node adalah time series 30 time step × 5 fitur. LSTM diproses **per node secara paralel** (weight sharing), menghasilkan hidden state 64-dimensi per node.

2. **Detection head**: hidden state semua node di-pool dengan **mean pooling** menjadi vektor 64-dimensi, lalu melewati MLP (64 → 32 → 2) untuk klasifikasi biner (normal vs fault).

3. **Localization head**: hidden state setiap node melewati MLP (64 → 32 → 1) untuk menghasilkan logit per node, yang merepresentasikan probabilitas node tersebut menjadi sumber fault.

Total parameter: **55.715**. Model ini ringan dan cocok untuk deployment edge.

#### 4.2 Loss Function

Loss total adalah kombinasi linear dari dua komponen:

\[
\mathcal{L} = \mathcal{L}_{det} + \lambda_{loc} \cdot \mathcal{L}_{loc}
\]

dengan \( \lambda_{loc} = 0.5 \).

- \( \mathcal{L}_{det} \): Cross-Entropy Loss dengan **class weights** untuk menangani ketidakseimbangan kelas (30% positif). Bobot yang dihitung: `neg = 0.716`, `pos = 1.660`.
- \( \mathcal{L}_{loc} \): Binary Cross-Entropy with Logits, dihitung per node.

#### 4.3 Optimizer dan Schedule

- **Optimizer**: AdamW dengan learning rate \( 1 \times 10^{-3} \) dan weight decay \( 1 \times 10^{-5} \).
- **Scheduler**: Cosine Annealing dengan \( T_{max} = 30 \) epoch.
- **Gradient clipping**: max norm 1.0 untuk stabilitas.
- **Batch size**: 128.
- **Epochs**: 30.

---

### 5. Hasil Training

#### 5.1 Training Dynamics

| Epoch | Train Loss | Val Loss | Val Acc | Val F1 | Val Prec | Val Rec |
|---|---|---|---|---|---|---|
| 1 | 0,8413 | 0,6668 | 0,7904 | 0,4804 | 0,7926 | 0,3447 |
| 5 | 0,6430 | 0,5907 | 0,8203 | 0,5423 | 0,9545 | 0,3788 |
| 10 | 0,4800 | 0,4305 | 0,8676 | 0,7232 | 0,8771 | 0,6152 |
| 15 | 0,4346 | 0,4294 | 0,8276 | 0,6976 | 0,6881 | 0,7074 |
| 20 | 0,4092 | 0,4358 | 0,8006 | 0,6805 | 0,6190 | 0,7555 |
| 25 | 0,3946 | 0,4207 | 0,8282 | 0,6965 | 0,6917 | 0,7014 |
| **30** | **0,3861** | **0,4307** | **0,8231** | **0,6981** | **0,6710** | **0,7275** |

**Observasi:**

1. **Train loss terus turun** dari 0,8413 ke 0,3861 (turun 54%) tanpa tanda-tanda plateau. Ini adalah indikasi awal **overfitting**: model menghafal training set.

2. **Val loss plateau** di sekitar 0,42–0,43 sejak epoch 12, dengan osilasi. Gap antara train loss (0,386) dan val loss (0,431) sebesar 0,045 adalah tanda overfitting moderat.

3. **Val F1 mencapai puncak 0,7624 pada epoch 18**, tetapi turun kembali ke 0,6981 pada epoch 30. Ini menunjukkan **instabilitas training**: model terbaik tidak berada di epoch terakhir.

4. **Val Precision dan Recall saling bertukar**: pada epoch awal, precision tinggi (0,95) tapi recall rendah (0,35). Pada epoch akhir, keduanya seimbang di sekitar 0,67–0,73. Model belajar untuk lebih agresif memprediksi fault, tetapi dengan konsekuensi false positive meningkat.

5. **Val accuracy** relatif stabil di kisaran 0,79–0,89, dengan puncak 0,8862 pada epoch 18.

#### 5.2 Metrik Test

Evaluasi pada test set (2.059 window, 617 fault aktif) menggunakan model dengan **F1 validasi terbaik** (epoch 18):

| Metrik | Nilai |
|---|---|
| **Accuracy** | **0,8747** |
| **Precision** | **0,9521** |
| **Recall** | **0,6126** |
| **F1-Score** | **0,7456** |
| TP | 378 |
| FP | 19 |
| FN | 239 |
| TN | 1.423 |

**Analisis:**

- **Precision 95,21%** berarti ketika model memprediksi fault, 95 kali dari 100 prediksi benar. Ini sangat baik untuk aplikasi yang ingin meminimalkan false alarm.
- **Recall 61,26%** berarti model melewatkan **38,74% fault** yang sebenarnya terjadi (239 dari 617). Ini masalah serius untuk sistem deteksi dini: hampir 4 dari 10 fault tidak terdeteksi.
- **Asimetri precision-recall** ini khas untuk model yang dilatih dengan class weight moderat dan threshold 0,5. Model cenderung **konservatif**: lebih memilih diam daripada salah alarm.

**Confusion matrix dalam bentuk tabel:**

|  | Predicted Normal | Predicted Fault |
|---|---|---|
| **Actual Normal** | 1.423 (TN) | 19 (FP) |
| **Actual Fault** | 239 (FN) | 378 (TP) |

False alarm rate: \( 19 / (19 + 1423) = 1,3\% \) — sangat rendah, bagus untuk operasional (operator tidak akan terganggu oleh alarm palsu).
Miss rate: \( 239 / (239 + 378) = 38,7\% \) — terlalu tinggi untuk sistem keselamatan.

#### 5.3 Performa per Jenis Skenario

Distribusi window test per jenis skenario:

| Jenis Skenario | Jumlah Window | Window Fault Aktif | Persentase |
|---|---|---|---|
| Normal | 355 | 0 | 0,0% |
| LLF | 568 | 205 | 36,1% |
| LGF | 568 | 208 | 36,6% |
| PSC | 568 | 204 | 35,9% |

Distribusi fault yang hampir seimbang antar jenis (LLF, LGF, PSC) menunjukkan bahwa model tidak bias terhadap jenis fault tertentu. Namun, karena kita tidak menyimpan prediksi per window dalam output ini, analisis lebih dalam (apakah model lebih baik mendeteksi LLF vs LGF) perlu dilakukan di Hari 4 dengan evaluasi per-kelas.

#### 5.4 Performa Lokalisasi

Dari 617 window yang benar-benar fault, model menghasilkan prediksi lokalisasi node:

| Metrik | Nilai | Baseline Random |
|---|---|---|
| **Top-1 Accuracy** | **0,3290** | 0,083 (1/12) |
| **Top-3 Accuracy** | **0,5964** | 0,250 (3/12) |
| **Mean IoU** | **0,1394** | — |

**Interpretasi:**

- **Top-1 accuracy 32,9%** berarti model menempatkan node fault sebenarnya di peringkat pertama probabilitas untuk 1 dari 3 kasus. Ini **4× lebih baik dari random** (8,3%), tetapi masih jauh dari memadai untuk aplikasi praktis.
- **Top-3 accuracy 59,6%** berarti node fault sebenarnya ada di 3 peringkat teratas untuk ~60% kasus. Ini lebih baik, tetapi masih ada 40% kasus di mana node fault tidak masuk top-3.
- **Mean IoU 13,94%** sangat rendah. Model cenderung memprediksi terlalu banyak atau terlalu sedikit node sebagai fault, dengan sedikit overlap dengan ground truth.

**Diagnosis:** Tanpa informasi spasial eksplisit (adjacency matrix), LSTM hanya dapat belajar lokalisasi dari pola temporal per node. Karena setiap node melihat time series-nya sendiri secara independen, model tidak dapat membedakan antara "node 3 fault" dan "node 7 fault" jika pola temporalnya mirip. Inilah **motivasi utama untuk menambahkan GAT** di Hari 4.

---

### 6. Diskusi

#### 6.1 Overfitting Moderat

Gap antara train loss (0,386) dan val loss (0,431) menunjukkan overfitting moderat. Model menghafal training set, terbukti dari train loss yang terus turun tanpa plateau. Beberapa faktor yang berkontribusi:

1. **Ukuran dataset terbatas**: 126 skenario training untuk 55.715 parameter adalah rasio yang tidak ideal. Setiap parameter "melihat" hanya ~160 window training.
2. **Regularisasi belum optimal**: dropout 0.2 mungkin kurang agresif untuk dataset sekecil ini. Weight decay \( 10^{-5} \) juga sangat kecil.
3. **Arsitektur tanpa information bottleneck**: LSTM dengan hidden 64 mampu mengkodekan banyak informasi spesifik training set.

**Rekomendasi untuk Hari 4:** tambahkan dropout ke 0.3, naikkan weight decay ke \( 10^{-4} \), dan gunakan early stopping berdasarkan val loss (bukan val F1).

#### 6.2 Precision-Recall Trade-off

Model dengan precision 95% dan recall 61% berada di **titik konservatif** dari kurva precision-recall. Untuk aplikasi deteksi fault, trade-off ini bisa disesuaikan:

- **Jika prioritas adalah keselamatan** (jangan sampai fault terlewat): turunkan threshold klasifikasi dari 0,5 ke 0,35. Ini akan meningkatkan recall dengan konsekuensi precision turun. Berdasarkan distribusi val, threshold 0,35 mungkin menghasilkan recall ~80% dengan precision ~75%.
- **Jika prioritas adalah operasional** (jangan sampai alarm palsu): pertahankan threshold 0,5. Model saat ini sudah baik untuk ini.

Untuk sistem deteksi fault PLTS, **prioritas keselamatan** seharusnya lebih tinggi. Rekomendasi: threshold 0,35–0,40.

#### 6.3 Lokalisasi Lemah

Top-1 accuracy 32,9% adalah hasil yang lemah, meskipun lebih baik dari random. Ada tiga penyebab:

1. **Tanpa informasi graf**: Setiap node dianalisis secara independen. Model tidak tahu bahwa node 0 dan node 1 dalam string yang sama terhubung secara seri, sehingga fault di node 0 memengaruhi node 1.
2. **Label lokalisasi jarang**: Hanya 617 window dengan label lokalisasi, dan setiap window hanya memiliki 1–2 node positif dari 12. Ini adalah **extreme class imbalance** di level node.
3. **Loss lokalisasi terlalu kecil**: \( \lambda_{loc} = 0.5 \) mungkin tidak cukup untuk memaksa model belajar lokalisasi dengan baik. Model lebih fokus pada deteksi karena loss detection lebih dominan.

**Rekomendasi untuk Hari 4:** GAT akan memberikan informasi graf ke model, dan \( \lambda_{loc} \) bisa dinaikkan ke 1.0 atau 1.5 untuk menyeimbangkan kedua task.

#### 6.4 Baseline untuk Perbandingan

Nilai baseline ini penting untuk mengukur peningkatan di Hari 4:

| Metrik | Baseline LSTM | Target GAT+PINN |
|---|---|---|
| Test Accuracy | 0,8747 | > 0,90 |
| Test F1 | 0,7456 | > 0,82 |
| Test Recall | 0,6126 | > 0,75 |
| Top-1 Localization | 0,3290 | > 0,55 |
| Top-3 Localization | 0,5964 | > 0,80 |
| Mean IoU | 0,1394 | > 0,30 |

Target ini realistis berdasarkan literatur: GAT-based fault detection pada array PV mencapai akurasi 96,8%, dan physics-informed learning dapat meningkatkan generalisasi pada data terbatas.

---

### 7. Keterbatasan

1. **Tanpa augmentasi data.** Training hanya menggunakan 126 skenario, tanpa augmentasi seperti penambahan noise, time warping, atau mixup. Untuk dataset kecil seperti ini, augmentasi bisa meningkatkan generalisasi secara signifikan.

2. **Threshold klasifikasi tetap 0,5.** Tidak ada pencarian threshold optimal pada validation set. Untuk aplikasi keselamatan, threshold yang lebih rendah akan memberikan recall yang lebih tinggi.

3. **Lokalisasi tanpa graf.** Baseline ini tidak menggunakan adjacency matrix yang sudah dibangun. Ini adalah by design — GAT akan ditambahkan di Hari 4.

4. **Tanpa physics constraint.** Model murni data-driven. Tidak ada penalti untuk prediksi yang melanggar hukum fisika (misalnya, arus negatif atau tegangan melebihi Voc).

5. **Evaluasi per-kelas fault belum dilakukan.** Kita tidak tahu apakah model lebih baik mendeteksi LLF, LGF, atau PSC. Analisis ini penting dan akan dilakukan di Hari 4.

6. **Tidak ada cross-validation.** Split tunggal (70/15/15) rentan terhadap variasi. K-fold cross-validation akan memberikan estimasi performa yang lebih robust, tetapi memakan waktu 5× lebih lama.

---

### 8. Referensi

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. (Arsitektur LSTM asli)

2. Kong, W., Dong, Z. Y., Jia, Y., Hill, D. J., Xu, Y., & Zhang, Y. (2019). Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural Network. *IEEE Transactions on Smart Grid*, 10(1), 841–851. (LSTM untuk time series energi)

3. Mellit, A., & Pavan, A. M. (2023). A Deep Learning-Based Approach for Fault Detection in Photovoltaic Systems. *Energy Conversion and Management*, 285, 116971. (LSTM untuk fault detection PV)

4. Hossain, M., Mekhilef, S., Danesh, M., Olatomiwa, L., & Shamshirband, S. (2018). Application of Extreme Learning Machine for Short Term Output Power Forecasting of Three Grid-Connected PV Systems. *Journal of Cleaner Production*, 167, 395–405. (Baseline data-driven untuk PV)

5. Jiang, W., Fu, X., Zhang, Y., Xiong, H., Wen, Y., & Guan, X. (2024). Anomaly Detection for Grid-Connected Photovoltaic Array via Graph Attention Mechanism. *Lecture Notes in Electrical Engineering*, 1179, 759–769. (GAT untuk PV, referensi untuk Hari 4)

6. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*. (Dasar teoretis untuk GNN, referensi untuk GAT)

7. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. *ICLR*. (Arsitektur GAT, referensi untuk Hari 4)

8. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. *Journal of Computational Physics*, 378, 686–707. (Dasar teoretis PINN, referensi untuk Hari 4)

9. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*. (Optimizer Adam)

10. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*. (AdamW, optimizer yang digunakan)

---

*Dokumen ini merupakan bagian dari studi "Desain Sistem Kontrol PLTS untuk Deteksi Dini dan Lokalisasi Fault dengan Data Terbatas" dan akan dilampirkan sebagai bagian metodologi dan hasil dalam laporan akhir.*

---

## Lampiran: Ringkasan Status Hari 3

| Output | Status | Catatan |
|---|---|---|
| `src/preprocessing.py` | ✅ Tersedia | Loading, normalisasi, sliding window, adjacency |
| `src/dataset.py` | ✅ Tersedia | PyTorch Dataset wrapper |
| `src/model_lstm.py` | ✅ Tersedia | Baseline LSTM dengan dua head |
| `scripts/train_baseline.py` | ✅ Tersedia | Pipeline end-to-end |
| `data/processed/train_val_test_split.npz` | ✅ Tersedia | 3,4 MB |
| `logs/lstm_baseline_metrics.json` | ✅ Tersedia | 5,9 KB |
| `figures/training_curves.png` | ✅ Tersedia | 133 KB |
| `figures/confusion_matrix_lstm.png` | ✅ Tersedia | 30 KB |
| **Test F1** | **0,7456** | Baseline untuk GAT+PINN |
| **Top-1 Localization** | **0,3290** | Perlu GAT untuk improved |
| **Training time** | **273,6 s** (CPU) | Reproducible |

**Prioritas sebelum Hari 4:**
1. Tambahkan GAT layer di antara LSTM encoder dan dua head (detection + localization).
2. Tambahkan physics residual loss sebagai komponen PINN.
3. Naikkan \( \lambda_{loc} \) ke 1,0–1,5 untuk menyeimbangkan deteksi dan lokalisasi.
4. Pertimbangkan threshold tuning pada validation set (0,35–0,45) untuk meningkatkan recall.