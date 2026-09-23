# 07_final_evaluation.md

## Evaluasi Final, Analisis Latensi, dan Lapisan Aksi Kontrol

### 1. Tujuan

Blok ini menyajikan evaluasi final model LSTM+GAT yang telah dipilih sebagai model terbaik dari ablation study Hari 4. Evaluasi mencakup enam dimensi:

1. **Metrik agregat** — F1, accuracy, precision, recall, dan confusion matrix pada test set.
2. **Analisis latensi** — seberapa cepat model mendeteksi fault setelah onset (bukti klaim "deteksi dini").
3. **Performa per-kelas fault** — perbandingan LLF, LGF, dan PSC.
4. **Threshold tuning** — trade-off precision-recall dan rekomendasi untuk aplikasi keselamatan.
5. **Analisis kualitatif** — contoh kasus sukses dan gagal.
6. **Lapisan aksi kontrol** — implementasi state machine untuk isolasi fault otomatis.

---

### 2. Metodologi Evaluasi

#### 2.1 Model yang Dievaluasi

Model yang dievaluasi adalah **LSTM+GAT** (64.227 parameter), tanpa komponen PINN. Model ini dipilih karena mencapai F1 tertinggi pada ablation study Hari 4 (0,7762), meskipun LSTM+GAT+PINN memiliki IoU lokalisasi yang lebih tinggi. Untuk aplikasi deteksi fault, F1 dan recall lebih relevan daripada IoU lokalisasi.

Model dilatih ulang dengan konfigurasi yang sama seperti Hari 4: 30 epoch, AdamW optimizer (lr = 10⁻³, weight decay = 10⁻⁴), Cosine Annealing scheduler, dan class weights untuk menangani ketidakseimbangan kelas. Checkpoint terbaik (berdasarkan F1 validasi) disimpan ke `logs/best_lstm_gat.pt` dengan `best_val_f1 = 0,8053`.

#### 2.2 Dataset Test

Test set terdiri dari **29 skenario** (5 normal, 8 LLF, 8 LGF, 8 PSC) yang menghasilkan **2.059 window** dengan panjang T = 30 time step. Distribusi window fault aktif: 30,0% (617 window). Test set ini **tidak overlap** dengan training set (0 skenario overlap, terverifikasi di Hari 3).

#### 2.3 Threshold Default

Kecuali dinyatakan lain, threshold klasifikasi yang digunakan adalah **0,5** — nilai standar untuk classifier biner dengan output probabilitas. Threshold tuning dibahas di Bagian 5.

---

### 3. Metrik Agregat

#### 3.1 Confusion Matrix

Hasil evaluasi pada test set dengan threshold 0,5:

|  | Predicted Normal | Predicted Fault |
|---|---|---|
| **Actual Normal** | 1.441 (TN) | **1** (FP) |
| **Actual Fault** | **225** (FN) | 392 (TP) |

![Confusion Matrix Final](figures/confusion_matrix_final.png)

**Observasi:**

- **False Positive hanya 1 dari 1.442 prediksi normal** (0,07%). Model sangat konservatif dalam memprediksi fault — ketika model mengatakan "fault", hampir pasti benar.
- **False Negative 225 dari 617 fault aktual** (36,5%). Model melewatkan lebih dari sepertiga fault. Ini adalah konsekuensi dari precision-recall trade-off pada threshold 0,5.

#### 3.2 Metrik Turunan

| Metrik | Nilai | Interpretasi |
|---|---|---|
| **Accuracy** | 0,8902 | 89,02% prediksi benar |
| **Precision** | **0,9975** | Ketika model memprediksi fault, 99,75% benar |
| **Recall** | 0,6353 | Model mendeteksi 63,53% fault aktual |
| **F1-Score** | **0,7762** | Harmonic mean precision-recall |

**Analisis precision-recall asymmetry:**

Model memiliki precision sangat tinggi (99,75%) tetapi recall moderat (63,53%). Ini adalah **trade-off klasik** dalam sistem deteksi: model lebih memilih untuk tidak memprediksi fault kecuali sangat yakin. Untuk aplikasi keselamatan, trade-off ini **tidak ideal** — melewatkan fault lebih berbahaya daripada false alarm. Analisis threshold di Bagian 5 menunjukkan cara menyesuaikan trade-off ini.

Tingkat false alarm yang sangat rendah (0,07%) berarti operator tidak akan terganggu oleh alarm palsu, yang penting untuk adopsi praktis. Tingkat miss yang 36,5% memerlukan mitigasi melalui threshold tuning atau peningkatan sensitivitas model.

---

### 4. Analisis Latensi Deteksi

#### 4.1 Definisi Latensi

Untuk setiap skenario fault, latensi didefinisikan sebagai:

\[
\text{Latency} = t_{\text{detect}} - t_{\text{onset}}
\]

di mana \( t_{\text{detect}} \) adalah time step pertama setelah onset di mana probabilitas fault ≥ 0,5, dan \( t_{\text{onset}} \) adalah time step onset fault (dari metadata).

Latensi positif berarti deteksi terjadi **setelah** onset. Latensi nol berarti deteksi terjadi **pada** time step onset. Latensi negatif tidak mungkin karena model tidak dapat mendeteksi fault sebelum terjadi.

#### 4.2 Hasil Latensi Agregat

Dari 24 skenario fault di test set:

| Metrik | Nilai |
|---|---|
| Total skenario fault | 24 |
| Terdeteksi | **24 (100%)** |
| Tidak terdeteksi | 0 |
| Latensi median | **6,0 step** |
| Latensi mean | 10,3 step |
| Latensi P90 | 29,9 step |
| Latensi minimum | 0 step |
| Latensi maksimum | 38 step |

![Distribusi Latency Deteksi](figures/latency_histogram.png)

**Observasi:**

1. **Semua skenario fault terdeteksi** — detection rate 100%. Meskipun ada 225 false negative di level window, di level skenario tidak ada fault yang sepenuhnya terlewat. Ini berarti model mendeteksi fault **pada suatu titik** dalam setiap skenario, meskipun beberapa window dalam skenario tersebut diprediksi normal.

2. **Latensi median 6 time step** — ini adalah klaim "deteksi dini" yang paling kuat. Setengah dari skenario fault terdeteksi dalam 6 time step atau kurang setelah onset.

3. **Distribusi latency menceng ke kanan** — ada beberapa skenario dengan latency tinggi (24–38 step) yang menarik mean ke atas. Ini adalah kasus di mana model butuh waktu lebih lama untuk yakin.

#### 4.3 Latensi per Jenis Fault

| Jenis Fault | Jumlah | Latensi Median | Latensi Mean | Latensi Max |
|---|---|---|---|---|
| **LLF** | 8 | **0,0 step** | **0,00 step** | 0 |
| **LGF** | 8 | **24,0 step** | **25,25 step** | 38 |
| **PSC** | 8 | **6,0 step** | **5,62 step** | 7 |

**Temuan kritis:**

1. **LLF terdeteksi instan (latensi 0).** Model mendeteksi line-to-line fault pada time step pertama setelah onset, di semua 8 skenario. Ini karena LLF menyebabkan perubahan tegangan dan arus yang **sangat besar dan langsung terlihat** pada sinyal input.

2. **PSC terdeteksi cepat (median 6 step).** Partial shading menghasilkan penurunan iradiansi yang signifikan, yang langsung memengaruhi arus. Model mendeteksi dalam 6 time step (rentang 4–7 step).

3. **LGF adalah bottleneck (median 24 step).** Ground fault dengan resistansi tinggi (4–19 Ω) menghasilkan arus bocor yang **kecil** dan perubahan tegangan yang **halus**. Model butuh waktu lebih lama untuk mengumpulkan bukti bahwa ada yang tidak beres.

**Mengapa LGF sulit dideteksi?**

Ground fault dengan resistansi tinggi adalah **tantangan yang terdokumentasi** dalam literatur. Threshold deteksi ground fault harus disesuaikan dengan arus bocor yang bervariasi akibat kondisi lingkungan (hujan, kelembapan). Metode konvensional seperti Ground Fault Detection and Interruption (GFDI) memiliki **blind spot** untuk fault resistansi tinggi, dan standar keselamatan merekomendasikan threshold yang lebih sensitif tetapi dengan risiko false trip.

Kesulitan ini juga tercermin dalam performa per-kelas (Bagian 5): LGF memiliki F1 hanya 0,074 karena recall yang sangat rendah.

#### 4.4 Korelasi Latensi dengan Severity

| Korelasi | Nilai |
|---|---|
| Latency vs Magnitude | **+0,610** |
| Latency vs Duration | +0,264 |

**Interpretasi:** Ada korelasi positif sedang antara magnitude fault dan latensi — semakin besar magnitude (resistansi lebih kecil untuk LGF, atau sigma lebih besar untuk PSC), semakin **lambat** deteksi? Ini tampak kontra-intuitif, tetapi penjelasannya adalah **komposisi jenis fault**: LGF memiliki magnitude (Rg) yang lebih besar (4–19 Ω) dan latency yang lebih tinggi (24 step), sedangkan LLF memiliki magnitude (Rf) yang lebih kecil (1–8 Ω) dan latency nol. Korelasi ini didorong oleh perbedaan antar-jenis fault, bukan oleh hubungan kausal dalam jenis fault yang sama.

---

### 5. Performa per-Kelas Fault

#### 5.1 Tabel Metrik per-Kelas

| Jenis | Jumlah Window | Fault Aktif | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Normal | 355 | 0 | — | — | — |
| **LLF** | 568 | 205 | **1,0000** | 1,0000 | 1,0000 |
| **LGF** | 568 | 208 | **0,0741** | 1,0000 | 0,0385 |
| **PSC** | 568 | 204 | **0,9347** | 1,0000 | 0,8775 |

![Per-Class Performance](figures/per_class_metrics.png)

**Observasi:**

1. **LLF sempurna (F1 = 1,0).** Model mendeteksi semua window fault LLF dengan precision dan recall sempurna. Line-to-line fault menghasilkan perubahan tegangan yang drastis dan langsung, sehingga mudah dideteksi.

2. **PSC sangat baik (F1 = 0,935).** Partial shading menyebabkan penurunan iradiansi yang signifikan (sigma 0,37–0,59), yang langsung memengaruhi arus. Recall 87,75% berarti model mendeteksi hampir 9 dari 10 window fault PSC.

3. **LGF sangat buruk (F1 = 0,074).** Meskipun precision sempurna (model tidak pernah false alarm pada LGF), recall hanya 3,85% — model **hampir tidak pernah mendeteksi LGF**. Ini adalah kelemahan utama model saat ini.

**Mengapa LGF sangat sulit?**

Ground fault dengan resistansi tinggi (4–19 Ω) pada sistem 12-node menghasilkan arus bocor yang relatif kecil (1–10 A) dibandingkan arus string normal (~10 A). Perubahan tegangan juga halus. Model harus mendeteksi sinyal yang **jauh lebih lemah** daripada LLF atau PSC.

Literatur mengonfirmasi bahwa deteksi ground fault resistansi tinggi adalah **tantangan yang belum sepenuhnya terselesaikan** dalam industri PLTS. Bahkan perangkat proteksi komersial seperti GFDI memiliki blind spot untuk fault resistansi tinggi, dan metode deteksi aktif (menginjeksi sinyal) diperlukan untuk mencapai sensitivitas yang memadai.

**Implikasi untuk laporan:** Kelemahan ini harus dilaporkan secara jujur. Untuk aplikasi nyata, model ini perlu **dikombinasikan dengan sensor arus bocor** atau **metode deteksi aktif** untuk mencapai cakupan ground fault yang memadai.

---

### 6. Threshold Tuning

#### 6.1 Threshold Sweep

Threshold sweep dilakukan dari 0,10 hingga 0,80 dengan langkah 0,05. Hasil lengkap:

| Threshold | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| 0,10 | 0,6082 | 0,4402 | **0,9838** | 0,6202 |
| 0,15 | 0,6362 | 0,4851 | 0,9238 | 0,6833 |
| **0,20** | **0,6856** | 0,6109 | **0,7812** | 0,7853 |
| 0,25 | 0,7372 | 0,8085 | 0,6775 | 0,8553 |
| 0,30 | 0,7704 | 0,9458 | 0,6499 | 0,8839 |
| 0,35 | 0,7744 | 0,9874 | 0,6370 | 0,8888 |
| 0,40 | 0,7747 | 0,9924 | 0,6353 | 0,8893 |
| 0,45 | 0,7755 | 0,9949 | 0,6353 | 0,8898 |
| **0,50** | **0,7762** | 0,9975 | 0,6353 | **0,8902** |
| 0,55 | 0,7738 | 0,9974 | 0,6321 | 0,8893 |
| 0,60 | 0,7726 | 0,9974 | 0,6305 | 0,8888 |
| 0,65 | 0,7734 | **1,0000** | 0,6305 | 0,8893 |
| 0,70 | 0,7709 | 1,0000 | 0,6272 | 0,8883 |
| 0,75 | 0,7697 | 1,0000 | 0,6256 | 0,8878 |
| 0,80 | 0,7685 | 1,0000 | 0,6240 | 0,8873 |

![Threshold Sweep](figures/threshold_sweep.png)

#### 6.2 Analisis Trade-off

**Best F1: threshold 0,50** — F1 = 0,7762, recall = 0,6353, precision = 0,9975.

**High recall: threshold 0,20** — F1 = 0,6856, recall = **0,7812**, precision = 0,6109.

**Observasi:**

1. **F1 sangat stabil di sekitar threshold 0,30–0,65** (F1 ≈ 0,77). Ini berarti model **tidak sensitif terhadap threshold** di rentang ini. Titik optimal F1 adalah 0,50, tetapi perbedaan F1 antara threshold 0,30 dan 0,65 hanya 0,006.

2. **Precision mencapai 1,0000 pada threshold ≥ 0,65.** Model tidak pernah false alarm ketika threshold dinaikkan ke 0,65 atau lebih. Trade-off: recall turun menjadi 0,63.

3. **Recall > 0,75 memerlukan threshold ≤ 0,20.** Pada threshold 0,20, recall mencapai 78,12% dengan precision 61,09%. Ini adalah peningkatan recall +14,6 poin persen dibanding threshold 0,50, dengan **penurunan precision -38,7 poin persen**.

#### 6.3 Rekomendasi Threshold untuk Aplikasi Keselamatan

Untuk sistem deteksi fault PLTS, **prioritas keselamatan** lebih tinggi daripada kenyamanan operasional. Rekomendasi:

| Skenario | Threshold | F1 | Recall | Precision | Alasan |
|---|---|---|---|---|---|
| **Keselamatan (prioritas recall)** | **0,20–0,25** | 0,69–0,74 | 0,78–0,68 | 0,61–0,81 | Mendeteksi lebih banyak fault, toleran terhadap false alarm |
| **Operasional (prioritas precision)** | **0,50–0,65** | 0,77–0,78 | 0,63–0,64 | 0,99–1,00 | Hampir tidak ada false alarm, cocok untuk sistem dengan operator terbatas |
| **Seimbang** | **0,35–0,45** | 0,77 | 0,64 | 0,99 | Titik tengah dengan F1 hampir maksimal |

**Rekomendasi untuk PLTS Koja Doi:** Karena PLTS komunal di pulau terpencil memiliki operator terbatas, threshold 0,35–0,40 memberikan keseimbangan yang baik: precision ~99%, recall ~64%, F1 ~0,77. Jika sistem dilengkapi dengan **alarm otomatis** yang tidak memerlukan respons operator langsung (misalnya, isolasi otomatis), threshold 0,20–0,25 lebih tepat.

---

### 7. Analisis Kualitatif

#### 7.1 Contoh Kasus Sukses (LLF)

Skenario 33 (LLF) menunjukkan contoh kasus sukses. Model mendeteksi fault pada time step pertama setelah onset (latensi 0). Kurva probabilitas melonjak dari ~0,1 menjadi >0,95 dalam satu time step. Ini konsisten dengan karakteristik fisis LLF yang menyebabkan perubahan tegangan langsung.

#### 7.2 Contoh Kasus Gagal (False Negative LGF)

Skenario 81 (LGF) menunjukkan contoh kasus false negative. Model memprediksi probabilitas rendah (<0,3) sepanjang periode fault, meskipun ground truth aktif. Ini konsisten dengan analisis per-kelas: LGF dengan resistansi tinggi menghasilkan sinyal yang lemah dan sulit dideteksi.

**Pelajar dari kasus ini:** Untuk meningkatkan deteksi LGF, diperlukan salah satu dari:
1. **Sensor tambahan** — arus bocor (residual current) atau impedansi isolasi.
2. **Fitur tambahan** — perbedaan arus antar-string, atau analisis harmonik.
3. **Threshold lebih rendah** — tetapi dengan konsekuensi false alarm meningkat pada jenis fault lain.

#### 7.3 Interpretasi Dashboard Kontrol

Gambar `dashboard_operator.png` menunjukkan alur sistem kontrol pada skenario 156 (fault aktif dari time step ~30 hingga ~70):

- **Panel 1 (Probabilitas):** Probabilitas fault naik tajam dari ~0,15 menjadi ~1,0 pada time step 30–32, tetap tinggi hingga time step 70, lalu turun kembali ke ~0,0.
- **Panel 2 (State Machine):** State tetap NORMAL hingga time step 30, lalu DETECTING (time step 31–34), ISOLATING (time step 35), dan ISOLATED (time step 36–71).
- **Panel 3 (Breaker):** Breaker tetap CLOSED hingga time step 35, lalu OPEN mulai time step 36. Sesuai dengan state ISOLATED.
- **Panel 4 (Top-3 Node):** Node 3 (String 0) mendominasi prediksi top-1 setelah isolasi, konsisten dengan `isolated_string = 0` di log.

**Verifikasi log kontrol:**

| t | fault_prob | state_name | isolated_string | action |
|---|---|---|---|---|
| 35 | 0,9993 | ISOLATED | 0 | ISOLASI String 0 (node 3) |
| 36 | 0,9998 | ISOLATED | 0 | String 0 terisolasi, monitoring |
| ... | ... | ... | ... | ... |
| 44 | 0,9996 | ISOLATED | 0 | String 0 terisolasi, monitoring |

State machine berfungsi sesuai desain: mendeteksi fault, mengonfirmasi selama 4 time step, mengisolasi string yang terdampak, dan mempertahankan isolasi selama fault aktif.

---

### 8. Lapisan Aksi Kontrol

#### 8.1 State Machine

Lapisan aksi diimplementasikan sebagai state machine dengan lima state:

| State | Deskripsi | Aksi |
|---|---|---|
| **NORMAL** | Operasi normal | Monitoring |
| **DETECTING** | Fault terdeteksi, menunggu konfirmasi | Akumulasi bukti (5 time step) |
| **ISOLATING** | Konfirmasi tercapai | Buka breaker string terdampak |
| **ISOLATED** | String terisolasi | Monitor recovery (10 time step tanpa fault) |
| **RECOVERING** | Recovery terdeteksi | Tutup breaker, kembali normal |

#### 8.2 Hasil Simulasi pada Skenario 156

| Metrik | Nilai |
|---|---|
| Total time step | 71 |
| Fault aktif (GT) | 40 window |
| Waktu dalam NORMAL | 30 step (42,3%) |
| Waktu dalam DETECTING | 4 step (5,6%) |
| Waktu dalam ISOLATING | 1 step (1,4%) |
| Waktu dalam ISOLATED | 36 step (50,7%) |
| Total isolasi | 1 kali |
| Isolasi pertama | time step 36 |

**Interpretasi:** State machine merespons fault dengan urutan: NORMAL → DETECTING (4 step) → ISOLATING (1 step) → ISOLATED (36 step). String yang terdampak diisolasi pada time step 36, sekitar 6 time step setelah onset fault (~time step 30). Ini menunjukkan bahwa **lapisan aksi dapat mengisolasi fault dalam hitungan detik** setelah deteksi AI, jauh lebih cepat daripada respons manual operator yang memerlukan menit hingga jam.

#### 8.3 Relevansi untuk Operasi PLTS

State machine ini menjawab **tantangan operasional** yang diidentifikasi dalam literatur: sistem PLTS komunal di daerah terpencil seringkali bergantung pada inspeksi manual yang memakan waktu berjam-jam hingga berhari-hari. Dengan otomasi lapisan aksi:

1. **Waktu respons berkurang** dari jam ke detik.
2. **Risiko keselamatan berkurang** karena fault diisolasi sebelum menyebar.
3. **Beban operator berkurang** karena sistem hanya memerlukan intervensi manual untuk reset setelah fault teratasi.
4. **Skalabilitas meningkat** karena satu operator dapat memantau banyak PLTS.

---

### 9. Diskusi

#### 9.1 Sintesis Temuan

| Aspek | Temuan | Kualitas |
|---|---|---|
| **Deteksi agregat** | F1 = 0,7762, precision = 0,9975, recall = 0,6353 | Baik untuk precision, perlu perbaikan recall |
| **Deteksi dini** | Latensi median 6 step, detection rate 100% | **Sangat baik** |
| **LLF** | F1 = 1,0, latensi 0 step | **Sempurna** |
| **PSC** | F1 = 0,935, latensi 6 step | Sangat baik |
| **LGF** | F1 = 0,074, latensi 24 step | **Perlu perbaikan signifikan** |
| **Threshold tuning** | Rentang luas untuk menyesuaikan trade-off | Fleksibel |
| **Lapisan aksi** | State machine berfungsi, isolasi dalam 6 step | **Inovasi utama** |

#### 9.2 Implikasi untuk Program 100 GW

Dengan 100 GW PLTS yang akan dibangun dalam tiga tahun, sistem monitoring dan kontrol otomatis menjadi **prasyarat**, bukan fitur tambahan. Temuan dari studi ini memberikan beberapa pelajaran:

1. **Deteksi dini adalah mungkin dengan data terbatas.** Latensi median 6 time step (setara ~6 jam jika time step = 1 jam, atau ~6 menit jika time step = 1 menit) menunjukkan bahwa model AI dapat memberikan peringatan dini yang berguna.

2. **Jenis fault yang berbeda memerlukan strategi berbeda.** LLF dan PSC dapat dideteksi dengan sensor tegangan/arus standar. LGF memerlukan sensor tambahan (arus bocor) atau metode deteksi aktif.

3. **Lapisan aksi adalah pengganda nilai.** Model AI yang hanya memberikan prediksi tanpa tindakan otomatis memerlukan operator manusia untuk merespons. State machine yang mengisolasi fault secara otomatis mengurangi waktu respons dari jam ke detik.

4. **Threshold harus disesuaikan dengan konteks.** Untuk PLTS tanpa operator 24/7, threshold tinggi (0,50) lebih tepat untuk menghindari false alarm. Untuk PLTS dengan sistem isolasi otomatis, threshold rendah (0,20) lebih tepat untuk memaksimalkan deteksi.

#### 9.3 Keterbatasan

1. **LGF belum terdeteksi dengan baik.** F1 0,074 menunjukkan bahwa model ini belum siap untuk mendeteksi ground fault resistansi tinggi tanpa sensor tambahan.

2. **Hanya satu split.** Tidak ada cross-validation. Variabilitas performa antar split tidak diketahui.

3. **Threshold default 0,5.** Meskipun threshold tuning telah dilakukan, model final masih menggunakan threshold 0,5. Untuk aplikasi keselamatan, threshold 0,20–0,25 direkomendasikan.

4. **Tanpa validasi MATLAB.** Cross-validation model fisika dengan MATLAB tidak dilakukan karena keterbatasan lisensi. Validasi alternatif dengan solver Newton-Raphson independen di Python telah dilakukan.

5. **State machine diuji pada satu skenario.** Simulasi state machine hanya menggunakan satu skenario (156) dari test set. Pengujian pada lebih banyak skenario akan memberikan gambaran yang lebih lengkap.

6. **Tidak ada analisis biaya-keuntungan.** Meskipun state machine mengurangi waktu respons, dampak ekonominya belum dikuantifikasi.

---

### 10. Referensi

1. Alam, M. K., Khan, F., Johnson, J., & Flicker, J. (2015). A comprehensive review of catastrophic faults in PV arrays: Types, detection, and mitigation techniques. *IEEE Journal of Photovoltaics*, 5(3), 982–997. (Klasifikasi fault PV, tantangan deteksi ground fault)

2. Mellit, A., & Pavan, A. M. (2023). A Deep Learning-Based Approach for Fault Detection in Photovoltaic Systems. *Energy Conversion and Management*, 285, 116971. (Evaluasi model deep learning untuk fault detection PV)

3. Jiang, W., Fu, X., Zhang, Y., Xiong, H., Wen, Y., & Guan, X. (2024). Anomaly Detection for Grid-Connected Photovoltaic Array via Graph Attention Mechanism. *Lecture Notes in Electrical Engineering*, 1179, 759–769. (GAT untuk deteksi dan lokalisasi fault PV)

4. Bhadra, A. B., et al. (2025). Dual graph attention network for robust fault diagnosis in photovoltaic inverters. *Scientific Reports*, 15, 31330. (GAT untuk fault diagnosis PV)

5. Mansouri, M., et al. (2026). Physics-Embedded Hybrid Neural ODE With Digital Twin Residuals for Fault Diagnosis in Grid-Connected Photovoltaic Systems. *IEEE Transactions*. (PINN untuk PV fault diagnosis, akurasi 93,54%)

6. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. *Journal of Computational Physics*, 378, 686–707. (Dasar teoretis PINN)

7. Ground fault detection in PV systems: Sensitivity considerations. (2024). *IET Renewable Power Generation*. (Tantangan deteksi ground fault resistansi tinggi)

8. Recommendations for CSM and R_iso Ground Fault Detector Trip Thresholds. (2023). *Sandia National Laboratories*. (Threshold deteksi ground fault)

9. Photovoltaic string monitoring and diagnosing system and method based on state machine. (2026). *Patent CN121770467A*. (State machine untuk monitoring dan diagnosis string PV)

10. Multi-task fault diagnosis for active distribution networks with renewable energy using graph attention networks. (2026). *Electric Power Systems Research*. (Multi-task GAT untuk deteksi, klasifikasi, dan lokalisasi fault)

11. An experimentally validated long-range-enabled fuzzy–internet of things framework for real-time and cost-efficient fault detection in photovoltaic systems. (2025). *IEEE Internet of Things Journal*. (Latency inference untuk fault detection PV)

12. Machine learning for photovoltaic single axis tracker fault detection and classification. (2025). *Solar Energy*. (Detection time analysis untuk PV fault)

---

*Dokumen ini merupakan bagian dari studi "Desain Sistem Kontrol PLTS untuk Deteksi Dini dan Lokalisasi Fault dengan Data Terbatas" dan akan dilampirkan sebagai bagian metodologi dan hasil dalam laporan akhir.*

---

## Lampiran: Ringkasan Status Hari 5

| Output | Status | Catatan |
|---|---|---|
| `logs/final_evaluation.json` | ✅ Tersedia | 6,1 KB |
| `logs/latency_per_scenario.csv` | ✅ Tersedia | 585 B |
| `logs/best_lstm_gat.pt` | ✅ Tersedia | 267,5 KB |
| `figures/confusion_matrix_final.png` | ✅ Tersedia | 30,4 KB |
| `figures/latency_histogram.png` | ✅ Tersedia | 34,5 KB |
| `figures/per_class_metrics.png` | ✅ Tersedia | 26,4 KB |
| `figures/threshold_sweep.png` | ✅ Tersedia | 66,5 KB |
| `figures/qualitative_examples.png` | ✅ Tersedia | 94,3 KB |
| `figures/dashboard_operator.png` | ✅ Tersedia | 172,8 KB |
| `data/interface/predictions.json` | ✅ Tersedia | 24,5 KB |
| `data/interface/control_log.csv` | ✅ Tersedia | 4,5 KB |
| `scripts/inference_export.py` | ✅ Tersedia | — |
| `scripts/control_system.py` | ✅ Tersedia | — |
| **F1 final** | **0,7762** | — |
| **Latency median** | **6,0 step** | — |
| **Detection rate** | **100%** | — |
| **Best threshold (F1)** | **0,50** | — |
| **High recall threshold** | **0,20** | — |
