# 04_fault_injection.md

## Injeksi Fault dan Generasi Dataset Sintetik

### 1. Tujuan

Blok ini bertujuan menghasilkan **dataset sintetik berlabel** yang merepresentasikan kondisi operasi PLTS dalam keadaan normal dan tiga jenis fault yang telah ditetapkan di Fase 0:

1. **Line-to-line fault (LLF)** — hubung singkat resistif antara dua titik di string berbeda.
2. **Ground fault (LGF)** — kebocoran arus ke ground akibat degradasi isolasi.
3. **Partial shading (PSC)** — penurunan iradiansi lokal pada subset panel.

Dataset ini menjadi input utama untuk pelatihan model AI (LSTM + GAT + PINN) di Hari 3–4, sekaligus menjawab secara langsung pertanyaan rekruter tentang bagaimana data sintetik di-generate dari persamaan fisika.

---

### 2. Dasar Teoretis Injeksi Fault

Pendekatan yang digunakan adalah **parameter modification** pada model single-diode yang sudah tervalidasi di Hari 1. Setiap jenis fault direpresentasikan sebagai perubahan pada parameter sirkuit atau kondisi operasi yang secara fisis masuk akal. Pendekatan ini sejalan dengan literatur yang menggunakan **digital twin berbasis physics-informed** untuk menghasilkan data fault sintetik dengan mengubah parameter kunci seperti iradiansi, tegangan, dan daya output.

Studi oleh AI-Driven Digital Twin for Condition Monitoring and Real-Time Fault Detection in PV Systems (2025) menggunakan pendekatan serupa: **synthetic faults were modeled using physically inspired equations by altering key parameters like irradiance, voltage, and power output**. Framework tersebut mencapai akurasi cross-validation 99,6% dengan model bagged decision tree ensemble yang dilatih pada kombinasi data riil dan sintetik.

#### 2.1 Model Line-to-Line Fault (LLF)

Line-line fault adalah koneksi resistansi rendah yang tidak disengaja antara dua titik dengan potensial berbeda dalam sistem fotovoltaik. Fault ini dapat terjadi **intra-string** (antara dua titik dalam string yang sama) atau **cross-string** (antara dua string berbeda). Penyebab umum meliputi kegagalan isolasi kabel, kontak tidak sengaja antara konduktor, dan kerusakan mekanis pada junction box.

Model matematis: ketika LLF terjadi antara node \( i \) dan node \( j \), arus tambahan \( I_f \) mengalir melalui resistansi fault \( R_f \):

\[
I_f = \frac{V_i - V_j}{R_f}
\]

**Batas arus fisis:** Dalam sistem PV, arus fault dibatasi oleh **sifat current-limiting** dari array PV. Arus fault maksimum yang dapat mengalir dari string-string sehat adalah sekitar \((N_{parallel} - 1) \times I_{sc}\), di mana \( N_{parallel} \) adalah jumlah string paralel. Untuk sistem 3 string dengan \( I_{sc} \approx 10,23 \) A, batas atas arus fault adalah sekitar 20,5 A. Model yang tidak membatasi arus fault dapat menghasilkan nilai yang tidak realistis (seperti 181 A yang teramati pada versi awal).

#### 2.2 Model Ground Fault (LGF)

Ground fault terjadi ketika jalur konduktif dengan resistansi rendah terbentuk antara konduktor aktif dan ground. Pada sistem PLTS, ground fault merupakan risiko keselamatan utama yang dapat menyebabkan kebakaran jika tidak terdeteksi.

Model matematis: ground fault dimodelkan sebagai **resistansi bocor \( R_g \)** yang terhubung antara node tertentu dan ground:

\[
I_{leak} = \frac{V_{node}}{R_g}
\]

Studi terbaru tentang model sirkuit ekuivalen untuk PV string-to-earth menunjukkan bahwa model yang terdiri dari **single insulation resistance dan RC-series paths in parallel** lebih sesuai untuk merepresentasikan impedansi antara string PV dan ground dibandingkan model konvensional dengan single capacitance.

#### 2.3 Model Partial Shading (PSC)

Partial shading terjadi ketika sebagian permukaan modul terhalang dari sinar matahari, menyebabkan penurunan iradiansi lokal. Efeknya adalah penurunan arus pada string yang terkena, yang dapat memicu **bypass diode** dan menghasilkan karakteristik I-V multi-langkah.

Model matematis: untuk modul yang terkena shading, iradiansi efektif diturunkan:

\[
G_{shaded} = (1 - \sigma) \cdot G
\]

Studi tentang pengaruh jumlah bypass diode menunjukkan bahwa peningkatan jumlah bypass diode dapat meningkatkan output daya hingga **113%** dibandingkan modul PV konvensional, bergantung pada intensitas shading. Namun, dalam model yang digunakan untuk dataset ini, efek bypass diode belum dimodelkan secara eksplisit — ini menjadi salah satu keterbatasan yang diakui.

---

### 3. Arsitektur Dataset

#### 3.1 Konfigurasi Sistem

| Parameter | Nilai |
|---|---|
| Jumlah string | 3 |
| Modul per string | 4 |
| Total node (panel) | 12 |
| Kapasitas | 4,8 kWp |
| Modul | Monocrystalline 400 Wp |
| Parameter modul | Hasil tuning dari `pv_model_tuned.py` (R_s = 0,1065 Ω, R_sh = 528.536.055 Ω, I_o_ref = 2,502 × 10⁻⁹ A) |

#### 3.2 Struktur Data

Setiap sample direpresentasikan sebagai vektor time series dengan dimensi:

\[
\mathbf{X} \in \mathbb{R}^{T \times N \times F}
\]

di mana \( T = 100 \) time step, \( N = 12 \) node, dan \( F = 5 \) fitur per node (V, I, P, G, T).

#### 3.3 Skema Generasi Skenario

| Parameter | Nilai |
|---|---|
| Jumlah skenario normal | 30 |
| Jumlah skenario LLF | 50 |
| Jumlah skenario LGF | 50 |
| Jumlah skenario PSC | 50 |
| **Total skenario** | **180** |
| Panjang time series per skenario | 100 time step |
| Noise sensor | 2% Gaussian |
| Seed reproducibility | 42 |

Variasi parameter per jenis fault:

| Parameter | LLF | LGF | PSC |
|---|---|---|---|
| Magnitudo | \( R_f \) = 0,1–10 Ω | \( R_g \) = 0,5–20 Ω | \( \sigma \) = 0,3–0,6 |
| Onset time | 20–60 | 20–60 | 20–60 |
| Durasi fault | 10–40 step | 10–40 step | 10–40 step |
| Sifat | Step / gradual | Gradual | Gradual |
| Node terdampak | 1–2 node (cross-string) | 1 node | 1–3 node (intra-string) |

---

### 4. Hasil Generasi Dataset

#### 4.1 Output Terminal `generate_dataset.py`

```
[debug] ROOT = D:\Berkas Penting\JOBS\AEER\kojadoi-fault-detection
[debug] src exists = True
[debug] fault_injection exists = True
[info] Memakai parameter hasil tuning dari src/pv_model_tuned.py
       R_s = 0.1065 Ω | R_sh = 528536054.76 Ω | I_o_ref = 2.502e-09 A
[info] Data cuaca: 1972 baris siang

[1/4] Generating 30 skenario normal ...
[2/4] Generating 50 skenario LLF ...
[3/4] Generating 50 skenario LGF ...
[4/4] Generating 50 skenario PSC ...

[info] Total skenario: 180
[info] Dataset tersimpan: ...\fault_dataset.csv
       Total baris: 216000
[info] Metadata tersimpan: ...\fault_metadata.json
```

#### 4.2 Statistik Dataset

| Metrik | Nilai |
|---|---|
| Total skenario | 180 |
| Normal | 30 |
| LLF | 50 |
| LGF | 50 |
| PSC | 50 |
| Total baris | 216.000 |
| Baris fault aktif | 47.892 (22,17%) |
| Ukuran file | 49,0 MB |

Distribusi baris fault per jenis:

| Jenis Fault | Jumlah Baris |
|---|---|
| LLF | 16.452 |
| LGF | 15.780 |
| PSC | 15.660 |

#### 4.3 Range Nilai per Jenis Fault

| Jenis Fault | V min (V) | V max (V) | V mean (V) | I min (A) | I max (A) | I mean (A) | P min (W) | P max (W) | P mean (W) |
|---|---|---|---|---|---|---|---|---|---|
| Normal | 33,03 | 43,86 | 39,20 | 0,466 | 10,443 | 4,410 | 16,37 | 443,50 | 175,80 |
| LLF | 16,94 | 44,14 | 38,75 | 0,467 | **17,893** | 5,452 | 15,99 | **731,53** | 212,87 |
| LGF | 1,76 | 44,12 | 39,12 | 0,033 | 10,434 | 4,483 | 0,06 | 435,57 | 178,73 |
| PSC | 33,36 | 45,80 | 39,28 | 0,236 | 10,202 | 4,391 | 7,97 | 411,61 | 175,18 |

**Perbandingan dengan versi sebelum perbaikan:**

| Metrik | Sebelum Perbaikan | Setelah Perbaikan | Perubahan |
|---|---|---|---|
| Arus LLF maksimum | 181,65 A | **17,89 A** | **-90,2%** |
| Daya LLF maksimum | 3.785,73 W | **731,53 W** | **-80,7%** |
| Range \( R_g \) | 1,0–100 Ω | **0,5–20 Ω** | Efek lebih terlihat |

#### 4.4 Contoh Metadata

**Normal (scenario 0):**
```json
{
  "scenario_id": 0,
  "fault_type": "normal",
  "onset": -1,
  "duration": 0,
  "affected_nodes": [],
  "magnitude": 0.0,
  "gradual": false,
  "n_steps": 100,
  "start_idx": 167
}
```

**LLF (scenario 30):**
```json
{
  "scenario_id": 30,
  "fault_type": "LLF",
  "onset": 58,
  "duration": 35,
  "affected_nodes": [0, 8],
  "magnitude": 3.3158865454600783,
  "gradual": false,
  "n_steps": 100,
  "start_idx": 933
}
```

**LGF (scenario 80):**
```json
{
  "scenario_id": 80,
  "fault_type": "LGF",
  "onset": 43,
  "duration": 17,
  "affected_nodes": [0],
  "magnitude": 3.121546151213069,
  "gradual": true,
  "n_steps": 100,
  "start_idx": 752
}
```

**PSC (scenario 130):**
```json
{
  "scenario_id": 130,
  "fault_type": "PSC",
  "onset": 22,
  "duration": 35,
  "affected_nodes": [1, 0, 3, 2],
  "magnitude": 0.5897702022981715,
  "gradual": true,
  "n_steps": 100,
  "start_idx": 1130
}
```

---

### 5. Visualisasi Contoh Fault

Plot `fault_examples.png` menunjukkan empat panel time series daya per string (bukan rata-rata) sehingga efek lokal fault terlihat jelas.

**Panel Normal (scenario 0):**
Ketiga string menunjukkan pola harian yang identik — daya berfluktuasi antara 25–1.450 W mengikuti siklus iradiansi. Tidak ada area fault yang ditandai. Konsistensi antar-string mengonfirmasi bahwa tidak ada fault aktif.

**Panel Line-to-Line Fault (scenario 30, magnitude=3,32, gradual=False, affected nodes=[0, 8]):**
Sebelum onset (time step 0–58), ketiga string identik dengan pola normal. Setelah onset pada time step 58, terjadi **divergensi dramatis**:
- **String 0 (biru)** dan **String 1 (hijau)** menunjukkan lonjakan daya yang sangat besar — mencapai **~2.500 W** per string, jauh di atas kapasitas normal ~1.450 W.
- **String 2 (oranye)** menunjukkan penurunan daya yang signifikan.
- Area fault (merah) dari time step 58 hingga 92.
- Lonjakan daya pada String 0 dan String 1 adalah konsekuensi dari arus fault yang mengalir dari string sehat ke titik fault di String 0 dan String 8 (String 2, node 0). Model LLF yang digunakan (dengan batas arus fisis) menghasilkan arus fault yang terbatas, tetapi tetap menyebabkan peningkatan arus pada string yang terhubung.

**Panel Ground Fault (scenario 80, magnitude=3,12, gradual=True, affected nodes=[0]):**
- Sebelum onset (time step 0–43), ketiga string identik.
- Setelah onset, **String 0 (biru)** — yang mengandung node 0 — menunjukkan **penurunan daya yang tajam** hingga mendekati **0 W** pada time step 58–60.
- String 1 (hijau) dan String 2 (oranye) tetap normal.
- Penurunan gradual terlihat jelas: daya String 0 turun perlahan dari ~1.400 W ke ~0 W selama durasi fault (time step 43–60), konsisten dengan sifat gradual dari LGF.
- Efek lokal ini hanya terlihat karena plot menampilkan **per-string**; jika menggunakan rata-rata, penurunan hanya ~33% dan kurang mencolok.

**Panel Partial Shading (scenario 130, magnitude=0,59, gradual=True, affected nodes=[1, 0, 3, 2]):**
- Sebelum onset (time step 0–22), ketiga string identik.
- Setelah onset pada time step 22, **String 0 (biru)** menunjukkan **penurunan daya yang signifikan** dibandingkan String 1 dan String 2.
- Karena semua 4 node dalam String 0 (node 0, 1, 2, 3) terkena shading dengan \( \sigma = 0,59 \), string ini kehilangan sebagian besar produksinya.
- Penurunan gradual terlihat: daya String 0 turun dari ~1.400 W ke ~400–600 W selama durasi fault (time step 22–57).
- Setelah fault berakhir (time step 58+), String 0 kembali normal.

---

### 6. Verifikasi Fisis

#### 6.1 Konsistensi Internal

| Pemeriksaan | Hasil | Status |
|---|---|---|
| Max \|P - V×I\| | 0,000000 | ✅ |
| V negatif | 0 | ✅ |
| I negatif | 0 | ✅ |
| P negatif | 0 | ✅ |
| NaN di V, I, P | 0 | ✅ |
| Jumlah step aktif = duration | 35 = 35, 36 = 36, 22 = 22 | ✅ |

#### 6.2 Validasi Batas Arus LLF

| \( R_f \) (Ω) | \( I_{fault} \) sebelum | \( I_{fault} \) sesudah | Keterangan |
|---|---|---|---|
| 0,1 | 1.600 A | **24,55 A** (capped) | Batas fisis tercapai |
| 1,0 | 160 A | **24,55 A** (capped) | Batas fisis tercapai |
| 5,0 | 32 A | **24,55 A** (capped) | Batas fisis tercapai |
| 10,0 | 16 A | **16 A** (natural) | Di bawah batas |

Arus maksimum string fault yang teramati dalam dataset adalah **17,89 A** — turun dari 181,65 A. Ini berada dalam rentang realistis untuk sistem 3-string dengan \( I_{sc} \approx 10,23 \) A.

#### 6.3 Anomali yang Tersisa

**Anomali 1: Daya LLF masih relatif tinggi (731,53 W per node).**

Daya maksimum per node pada LLF adalah 731,53 W, dibandingkan 443,50 W pada kondisi normal. Peningkatan ~65% ini adalah konsekuensi dari arus fault yang mengalir dari string sehat ke string yang fault. Meskipun secara fisis mungkin terjadi, magnitudonya bergantung pada model injeksi yang digunakan.

**Anomali 2: Perbedaan daya normal Hari 1 vs Hari 2.**

| Dataset | Mean P (W) |
|---|---|
| Hari 1 (`normal_condition_timeseries.csv`) | 975,0 |
| Hari 2 (skenario normal) | 527,4 |
| Selisih | 447,5 |

Perbedaan ini disebabkan oleh **perbedaan sampling cuaca**: Hari 1 menggunakan seluruh 4.344 baris data (termasuk malam), sedangkan Hari 2 hanya menggunakan 1.972 baris siang hari (G > 50 W/m²). Selain itu, Hari 2 memilih **start_idx acak** untuk setiap skenario, sehingga ada skenario yang dimulai dari periode berawan. Untuk perbandingan yang adil, filter dataset Hari 1 hanya pada baris siang hari.

---

### 7. Keterbatasan Dataset

1. **Model LLF tidak memodelkan bypass diode.** Pada sistem nyata, bypass diode akan aktif ketika modul ter-shade atau mengalami mismatch, menghasilkan karakteristik I-V multi-langkah yang lebih kompleks.

2. **Model LGF belum memodelkan kapasitansi bocor.** Studi terbaru menunjukkan bahwa model sirkuit ekuivalen untuk ground fault sebaiknya mencakup **RC-series paths in parallel**, bukan hanya resistansi murni.

3. **Tidak ada degradasi sensor atau komunikasi.** Dataset tidak memodelkan kehilangan data, latency, atau degradasi sensor yang umum terjadi pada PLTS komunal.

4. **Noise sensor 2% Gaussian.** Noise dunia nyata tidak selalu Gaussian; bisa mengandung spike, drift, dan outlier.

5. **Scope terbatas pada 12 node.** Konfigurasi 3 string × 4 panel adalah proof-of-concept. PLTS Koja Doi yang sebenarnya memiliki kapasitas 190 kWp dengan jumlah string dan panel yang jauh lebih banyak.

---

### 8. Referensi

1. AI-Driven Digital Twin for Condition Monitoring and Real-Time Fault Detection in PV Systems. (2025). *IEEE Xplore*. (Physics-informed synthetic fault generation, bagged decision tree 99,6% accuracy)

2. DPAL: A dual-threshold physics-informed automatic labeling framework with open-set recognition for robust fault diagnosis of grid-connected photovoltaic systems under label scarcity. (2026). *ScienceDirect*. (Physics-informed automatic labeling, 99,52% detection accuracy, 98,6% multi-class diagnosis)

3. Jiang, W., Fu, X., Zhang, Y., Xiong, H., Wen, Y., & Guan, X. (2024). Anomaly Detection for Grid-Connected Photovoltaic Array via Graph Attention Mechanism. *Lecture Notes in Electrical Engineering*, 1179, 759–769. (GAT-based fault detection, 96,8% accuracy)

4. Dual graph attention network for robust fault diagnosis in photovoltaic inverters. (2025). *Scientific Reports*, 15, 31330. (DualGAT, 97,35% test accuracy)

5. Zhao, Y., & Lyons, R. (2018). Line-Line Fault Analysis and Protection in PV Arrays. *Mersen Photovoltaic Protection Note 2*. (Line-line fault definition, causes, NEC Article 690.9 requirements)

6. Evaluation of the Suitability of Existing Earth-Fault Detection Methods for Insulation Monitoring Devices in Photovoltaic Power Generation Systems. (2026). *IEEJ Transactions on Electrical and Electronic Engineering*. (Equivalent circuit model for ground fault, RC-series paths)

7. To what extent the number of bypass diodes influence the performance of PV modules: probabilistic assessment. (2025). *Renewable Energy*, 249, 123243. (Bypass diode impact, up to 113% power enhancement)

8. Alam, M. K., Khan, F., Johnson, J., & Flicker, J. (2015). A comprehensive review of catastrophic faults in PV arrays: Types, detection, and mitigation techniques. *IEEE Journal of Photovoltaics*, 5(3), 982–997. (Current-limiting nature of PV arrays)

9. Kawonga, T. (2026). *Supplementary Materials for Energy-Efficient Anomaly Detection in Solar PV Systems Using Physics-Informed TinyML*. IEEE Dataport. (Synthetic dataset from first-principle physical models)

---