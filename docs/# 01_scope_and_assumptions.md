# 01_scope_and_assumptions.md

## Ruang Lingkup dan Asumsi

### Studi Kasus: Desain Sistem Kontrol PLTS untuk Deteksi Dini dan Lokalisasi Fault dengan Data Terbatas

**Proyek:** Power System & Energy Transition Analyst — AKSI EKOLOGI & EMANSIPASI RAKYAT (AEER)
**Periode Studi:** 6 hari (proof-of-concept)
**Penulis:** [Nama Anda]
**Tanggal:** [Tanggal]

---

### 1. Latar Belakang

#### 1.1 Profil Wilayah dan PLTS Koja Doi

**Gambaran Geografis Pulau Koja Doi**

Pulau Koja Doi merupakan salah satu pulau kecil yang terletak di Kecamatan Alok Timur, Kabupaten Sikka, Provinsi Nusa Tenggara Timur. Pulau ini berada pada koordinat sekitar **8,496° LS dan 122,399° BT**, dengan topografi pesisir dataran rendah dan elevasi rata-rata 10–50 meter di atas permukaan laut. Secara administratif, wilayah ini mencakup Desa Kojadoi yang terdiri dari beberapa dusun, yaitu Dusun Kojadoi, Dusun Koja Besar, dan Dusun Margajong.

![Gambaran lokasi dan tata letak PLTS Koja Doi](figures/Picture1.png)

*Gambar 1. Gambaran lokasi PLTS Koja Doi dan tata letak sistem di Pulau Koja Doi, Kecamatan Alok Timur, Kabupaten Sikka, Nusa Tenggara Timur. Sumber: dokumentasi lapangan.*

Akses ke Pulau Koja Doi hanya dapat ditempuh melalui transportasi laut dari Pelabuhan Maumere, dengan waktu tempuh sekitar 30–60 menit bergantung pada kondisi cuaca dan jenis perahu yang digunakan. Kondisi geografis kepulauan ini memberikan tantangan tersendiri dalam hal logistik, pemeliharaan infrastruktur, dan pengiriman suku cadang — faktor-faktor yang secara langsung memengaruhi strategi operasi dan pemeliharaan (O&M) PLTS.

**Karakteristik Sosio-Ekonomi Masyarakat**

Mayoritas penduduk Desa Kojadoi bekerja sebagai nelayan dan petani. Dari total 471 kepala keluarga (KK) di desa tersebut, sebelum keberadaan PLTS hanya sekitar 86 KK yang berlangganan listrik PLN. Keterbatasan akses listrik sebelum PLTS mendorong ketergantungan pada genset berbahan bakar minyak (BBM) dengan biaya operasional tinggi dan pasokan yang tidak stabil.

**Spesifikasi Teknis PLTS Koja Doi**

PLTS Koja Doi merupakan pembangkit listrik tenaga surya komunal dengan kapasitas **190 kWp** yang mulai beroperasi pada tahun 2019 dan dikelola oleh PLN Wilayah NTT. Sistem ini merupakan PLTS komunal **off-grid** yang dilengkapi dengan sistem penyimpanan baterai untuk menampung daya agar listrik dapat digunakan pada malam hari. Konfigurasi sistem mencakup:

| Komponen | Keterangan |
|---|---|
| Kapasitas terpasang | 190 kWp |
| Jenis modul | Monocrystalline |
| Konfigurasi | Off-grid dengan penyimpanan baterai |
| Pengelola | PLN Wilayah NTT |
| Tahun operasi | 2019 |
| Jumlah pelanggan | ~200 pelanggan (86 KK terdata awal, berkembang seiring waktu) |
| Wilayah layanan | Dusun Kojadoi dan Dusun Koja Besar |

Dari total 471 KK di desa, saat ini PLTS Koja Doi telah melayani sekitar 200 pelanggan. Meskipun demikian, masih terdapat satu dusun lain (Dusun Margajong) yang belum teraliri listrik, menunjukkan bahwa perluasan akses energi masih menjadi agenda yang belum selesai di wilayah ini.

**Dampak Sosial dan Ekonomi**

Kehadiran PLTS Koja Doi telah mengubah berbagai aspek kehidupan masyarakat setempat. Sebelumnya, desa kepulauan ini bergantung pada bahan bakar minyak (BBM) dengan ketersediaan listrik yang sangat terbatas. Sejak PLTS beroperasi, akses listrik yang lebih stabil telah:

1. **Mendorong pertumbuhan ekonomi lokal** — memungkinkan usaha kecil seperti kios, bengkel, dan pengolahan hasil laut untuk beroperasi lebih lama.
2. **Meningkatkan akses pendidikan** — anak-anak dapat belajar pada malam hari dengan penerangan yang memadai.
3. **Meningkatkan kualitas hidup** — akses terhadap penerangan, pendingin, dan peralatan rumah tangga modern.
4. **Mengurangi ketergantungan pada BBM** — penghematan biaya operasional dan pengurangan emisi karbon.

Namun, keberlanjutan manfaat ini sangat bergantung pada **keandalan operasional PLTS**. Seperti halnya sistem PLTS komunal di daerah terpencil lainnya di Indonesia, tantangan utama yang dihadapi adalah keterbatasan sistem monitoring, minimnya data historis, dan sulitnya akses teknisi terlatih. Hal ini menjadikan PLTS Koja Doi sebagai studi kasus yang representatif untuk pengembangan sistem kontrol berbasis kecerdasan buatan yang mampu mendeteksi fault secara dini dengan data terbatas.

#### 1.2 Relevansi dengan Program 100 GW

Presiden Prabowo Subianto telah mencanangkan program pembangunan PLTS dengan total kapasitas 100 GW dalam tiga tahun (2026–2029). Program ini terdiri dari 80 GW PLTS terdistribusi dan 20 GW PLTS skala utilitas, dengan total investasi yang diperkirakan mencapai US$73 miliar atau lebih dari Rp1.140 triliun. PLTS Koja Doi merupakan representasi mikro dari tantangan yang akan dihadapi dalam skala nasional: bagaimana memastikan keandalan operasional, mendeteksi gangguan secara dini, dan meminimalkan biaya pemeliharaan pada ribuan PLTS yang tersebar di seluruh Indonesia.

Sistem kontrol berbasis kecerdasan buatan (AI) yang dirancang dalam studi ini menawarkan solusi yang dapat direplikasi untuk mendukung program 100 GW, khususnya dalam konteks PLTS komunal di daerah terpencil yang memiliki keterbatasan data historis dan infrastruktur monitoring.


---

### 2. Batasan Studi

#### 2.1 Batasan Konfigurasi Sistem

Karena keterbatasan waktu yang ketat (6 hari), studi ini tidak menyimulasikan seluruh kapasitas 190 kWp PLTS Koja Doi. Sebagai gantinya, digunakan konfigurasi representatif skala kecil:

| Parameter | Nilai | Keterangan |
|---|---|---|
| Jumlah string paralel | 3 string | Representasi multi-string |
| Jumlah panel per string | 4 panel (seri) | Representasi string tipikal |
| Total node/panel | 12 node | Cukup untuk analisis spasial GAT |
| Kapasitas total | 4,8 kWp | Proof-of-concept |
| Tegangan string (STC) | ≈ 197,6 V | 4 × V_oc modul |
| Arus total (MPP) | ≈ 28,98 A | 3 × I_mpp modul |

**Justifikasi:** Konfigurasi 3 string × 4 panel dipilih karena memungkinkan analisis spasial antar-string dan antar-panel yang bermakna untuk Graph Attention Network (GAT), tanpa membengkakkan kompleksitas komputasi. Pendekatan ini sejalan dengan praktik umum dalam penelitian fault detection PLTS, di mana sistem skala laboratorium digunakan sebagai validasi awal sebelum penerapan skala penuh.

#### 2.2 Batasan Jenis Fault

Studi ini membatasi diri pada tiga jenis fault yang paling umum terjadi pada PLTS:

| Jenis Fault | Parameter yang Dimodifikasi | Rentang Magnitudo | Sifat Onset | Referensi |
|---|---|---|---|---|
| **Line-to-line (LLF)** | Resistansi hubung antara dua titik di string berbeda | 0,1–10 Ω | Step atau gradual | Othman & Al-Yozbaky (2024) |
| **Ground fault (LGF)** | Resistansi bocor ke ground pada node tertentu | 1–100 Ω | Gradual (degradasi isolasi) |核心 (Core.ac.uk) |
| **Partial shading (PSC)** | Iradiansi lokal pada subset panel | 40–70% dari iradiansi normal | Gradual (awan bergerak) | Patel et al. (2021) |

Pemilihan tiga jenis fault ini didasarkan pada frekuensi kejadiannya di lapangan dan ketersediaan referensi metodologis untuk injeksi sintetik. Line-to-line fault terjadi ketika koneksi resistansi rendah secara tidak sengaja menghubungkan dua titik dengan potensial berbeda dalam sistem fotovoltaik. Ground fault merupakan salah satu risiko keselamatan utama pada sistem PLTS yang dapat menyebabkan kebakaran jika tidak terdeteksi. Partial shading dapat disebabkan oleh awan, pohon, atau bangunan yang menghalangi sebagian permukaan modul.

#### 2.3 Batasan Platform Simulasi

Studi ini menggunakan Python sebagai tulang punggung komputasi, dengan dukungan pvlib untuk pemodelan fisika PV dan PyTorch untuk implementasi model AI. MATLAB hanya dijadikan alat validasi opsional jika waktu memungkinkan. Keputusan ini diambil untuk meminimalkan risiko debugging lintas platform dalam batas waktu yang ketat.

---

### 3. Justifikasi Batasan

#### 3.1 Keterbatasan Waktu

Studi ini dirancang untuk diselesaikan dalam 6 hari kerja. Dengan alokasi waktu tersebut, prioritas utama adalah menghasilkan pipeline yang berjalan end-to-end dan hasil yang masuk akal, bukan cakupan yang luas namun setengah jadi. Pemangkasan scope secara agresif dilakukan untuk memastikan setiap tahap (pemodelan fisika, injeksi fault, pelatihan AI, evaluasi) dapat diselesaikan dengan kualitas yang memadai.

#### 3.2 Data Terbatas sebagai Tantangan Inti

Tantangan utama yang diangkat dalam studi ini adalah keterbatasan data riil untuk pelatihan model AI. PLTS Koja Doi, seperti banyak PLTS komunal lainnya di Indonesia, tidak memiliki sistem monitoring terpusat dengan sensor berlapis. Data yang tersedia terbatas pada catatan produksi harian dan log pemeliharaan manual. Oleh karena itu, pendekatan yang diusulkan adalah membangun digital twin berbasis persamaan fisika yang mampu menghasilkan data sintetik berlabel.

Pendekatan ini sejalan dengan tren terkini dalam penelitian fault detection PLTS. DPAL (Dual-threshold Physics-informed Automatic Labeling) framework telah menunjukkan akurasi deteksi 99,52% pada kondisi label-scarce dengan memanfaatkan physics-informed automatic labeling. Physics-Embedded Hybrid Neural ODE (PE-HANODE) dengan Digital Twin residual juga telah terbukti meningkatkan akurasi klasifikasi fault hingga 93,54%, dibandingkan 21,99% untuk pendekatan Digital Twin murni.

#### 3.3 Prinsip Kejujuran Teknis

Studi ini secara eksplisit mengakui keterbatasan pendekatan data sintetik. Data sintetik tidak dapat menangkap semua ketidakpastian dunia nyata seperti degradasi sensor, gangguan komunikasi, dan variasi cuaca ekstrem lokal. Namun, dengan mengintegrasikan physics constraint ke dalam arsitektur AI (PINN), risiko overfitting terhadap noise sintetik dapat dikurangi. Rekomendasi untuk kalibrasi dengan data lapangan riil akan disertakan dalam diskusi keterbatasan.

---

### 4. Asumsi Data

#### 4.1 Sumber Data Iradiansi dan Suhu

Data iradiansi matahari dan suhu udara diperoleh dari **NASA POWER (Prediction of Worldwide Energy Resources)** API dengan spesifikasi berikut:

| Parameter | Nilai |
|---|---|
| API Endpoint | `https://power.larc.nasa.gov/api/temporal/hourly/point` |
| Parameter | `ALLSKY_SFC_SW_DWN` (iradiansi global horizontal, W/m²) dan `T2M` (suhu udara 2 m, °C) |
| Koordinat | Latitude: -8,496° ; Longitude: 122,399° |
| Periode | Januari–Juni 2025 (6 bulan) |
| Resolusi Temporal | Per jam |
| Komunitas | RE (Renewable Energy) |

**Justifikasi Pemilihan NASA POWER:** NASA POWER menyediakan data berbasis satelit dan reanalysis yang mencakup wilayah Indonesia dengan resolusi temporal per jam. Validasi data NASA POWER untuk radiasi matahari di wilayah tropis telah menunjukkan performa yang baik dengan korelasi tinggi dan kesalahan relatif kecil, yaitu <3,00 MJ/m²/hari. Studi validasi di Ghana dan wilayah tropis lainnya juga menunjukkan bahwa NASA POWER memiliki bias yang lebih kecil dibandingkan produk alternatif seperti ERA5-Land, dengan nilai KGE (Kling-Gupta Efficiency) mencapai 0,67.

**Catatan Konversi Waktu:** NASA POWER mengembalikan waktu dalam UTC. Karena PLTS Koja Doi berada di zona waktu WITA (UTC+8), data timestamp dikonversi ke waktu lokal dengan menambahkan 8 jam sebelum digunakan dalam simulasi. Konversi ini penting untuk memastikan pola harian iradiansi (puncak sekitar jam 12:00–13:00 waktu lokal) muncul dengan benar.

#### 4.2 Penanganan Missing Value

NASA POWER menggunakan nilai -999 untuk menandai data yang hilang. Dalam pra-pemrosesan, nilai -999 diganti dengan NaN dan diinterpolasi secara linear dengan `limit_direction='both'` untuk menjaga kontinuitas time series. Jumlah missing value setelah interpolasi dilaporkan dalam output script untuk verifikasi.

#### 4.3 Data Sintetik untuk Fault Injection

Data fault sintetik dihasilkan dengan menginjeksikan gangguan pada model fisika melalui modifikasi parameter. Label fault dihasilkan secara otomatis dari parameter injeksi, sehingga diperoleh dataset berlabel tanpa memerlukan anotasi manual. Pendekatan physics-informed automatic labeling serupa telah terbukti mencapai akurasi deteksi 99,52% dan akurasi diagnosis multi-kelas 98,6% pada dataset GPVS-Faults.

Untuk meningkatkan realisme, noise Gaussian dan non-Gaussian ditambahkan pada sinyal arus, tegangan, dan iradiansi. Ketidakpastian parameter (toleransi modul, degradasi baterai) dimodelkan menggunakan distribusi Monte Carlo.

---

### 5. Asumsi Fisika

#### 5.1 Model Sel Fotovoltaik

Model yang digunakan adalah **single-diode model** dengan parameter yang dihitung menggunakan **De Soto model** yang diimplementasikan dalam `pvlib.pvsystem.calcparams_desoto`. Model ini merepresentasikan karakteristik arus-tegangan (I-V) modul PV melalui persamaan:

$$I = I_{ph} - I_0\left[\exp\left(\frac{V + IR_s}{a}\right) - 1\right] - \frac{V + IR_s}{R_p}$$

di mana:
- $I_{ph}$ = arus fotogenerasi (A)
- $I_0$ = arus saturasi dioda (A)
- $R_s$ = resistansi seri (Ω)
- $R_p$ = resistansi paralel/shunt (Ω)
- $a$ = faktor idealitas dioda yang dimodifikasi ($n N_s V_{th}$)

Pendekatan De Soto telah divalidasi secara luas untuk modul fotovoltaik silikon kristalin. Studi terbaru menunjukkan bahwa model ini tidak menunjukkan bias sistematis yang signifikan dalam rekonstruksi arus, dengan nilai normalized mean bias error (nMBE) sebesar 0,02% untuk modul monofacial.

#### 5.2 Parameter Modul

Asumsi parameter modul monocrystalline 400 Wp yang digunakan:

| Parameter | Nilai | Satuan | Sumber |
|---|---|---|---|
| $P_{max}$ | 400 | Wp | Datasheet generik |
| $V_{oc}$ | 49,4 | V | Datasheet generik |
| $I_{sc}$ | 10,23 | A | Datasheet generik |
| $V_{mpp}$ | 41,4 | V | Datasheet generik |
| $I_{mpp}$ | 9,66 | A | Datasheet generik |
| $\alpha_{sc}$ | 0,0051 | A/°C | +0,05%/°C × $I_{sc}$ |
| $\beta_{voc}$ | -0,143 | V/°C | -0,29%/°C × $V_{oc}$ |
| $\gamma_{pdc}$ | -0,0034 | /°C | -0,34%/°C |
| $N_s$ | 72 | sel | Estimasi dari $V_{oc}$ |
| $n$ | 1,2 | — | Faktor idealitas tipikal |
| $R_s$ | 0,35 | Ω | Estimasi awal |
| $R_{sh}$ | 300 | Ω | Estimasi awal |

Parameter $R_s$ dan $R_{sh}$ diestimasi dari datasheet dan akan disesuaikan melalui validasi kurva I-V pada Hari 1 untuk memastikan error $P_{max}$ < 5%.

#### 5.3 Arsitektur AI yang Diusulkan

Sistem kontrol berbasis AI yang dirancang menggunakan tiga komponen utama:

| Komponen | Fungsi | Referensi |
|---|---|---|
| **LSTM** | Prediksi temporal, deteksi anomali berbasis residual | LSTM efektif untuk analisis time-series PV dan memiliki kemampuan pembelajaran pola temporal yang baik |
| **GAT** | Analisis spasial, lokalisasi fault antar-node | Graph Attention Network telah mencapai akurasi 96,8% untuk lokalisasi fault pada array PV |
| **PINN** | Konsistensi fisika, mitigasi overfitting pada data terbatas | Physics-informed learning telah terbukti meningkatkan generalisasi pada PLTS yang berbeda dari data pelatihan |

Kombinasi LSTM–GAT–PINN dipilih untuk menjawab tantangan data terbatas: LSTM menangkap dependensi temporal, GAT menangkap hubungan spasial antar-node, dan PINN memastikan prediksi model konsisten dengan hukum fisika yang berlaku.

#### 5.4 Asumsi Noise Sensor

Noise sensor diasumsikan Gaussian dengan standar deviasi 1–3% dari nilai sinyal. Asumsi ini didasarkan pada spesifikasi tipikal sensor tegangan dan arus yang digunakan pada sistem monitoring PLTS skala komunal. Noise ditambahkan pada semua sinyal V, I, dan P sebelum digunakan untuk pelatihan model AI.

#### 5.5 Asumsi Tanpa Degradasi Jangka Panjang

Studi ini tidak memodelkan degradasi jangka panjang modul PV (seperti PID atau LID) karena horizon waktu simulasi yang terbatas (6 bulan). Fokus studi adalah pada deteksi fault yang terjadi dalam skala waktu jam hingga hari, bukan degradasi bertahap selama bertahun-tahun.

---

### 6. Output yang Diharapkan

Pada akhir studi 6 hari, output yang akan dihasilkan:

| Output | Deskripsi | Hari |
|---|---|---|
| `koja_doi_irradiance_temperature_hourly.csv` | Data iradiansi dan suhu per jam dari NASA POWER | 1 |
| `normal_condition_timeseries.csv` | Time series kondisi normal (V, I, P) | 1 |
| `fault_dataset.csv` | Dataset sintetik berlabel dengan 3 jenis fault | 2 |
| Model LSTM + GAT + PINN | Arsitektur AI terlatih | 3–4 |
| Metrik evaluasi | Akurasi deteksi, latency, akurasi lokalisasi | 4–5 |
| Laporan akhir | Laporan lengkap dengan rekomendasi kebijakan 100 GW | 6 |

---

### 7. Referensi

1. Likadja, F. J. (2020). Sosialisasi Pengoperasian PLTS Off Grid Kojadoi Berkapasitas 190 KWP di Desa Kojadoi, Kecamatan Alok Timur, Kab. Sikka. *Jurnal Pengabdian Kepada Masyarakat*, 1(1).

2. De Soto, W., Klein, S. A., & Beckman, W. A. (2006). Improvement and validation of a model for photovoltaic array performance. *Solar Energy*, 80(1), 78–88.

3. Pradiko, I., et al. (2026). Kalibrasi dan Validasi Data Cuaca NASA POWER Menggunakan Data Pengamatan Stasiun Meteorologi Pertanian Khusus (SMPK) dan Automatic Weather Station (AWS). *Warta IOPRI*.

4. Liu, J., Huang, Y., Chen, K., Liu, G., Yan, J., Chen, S., Xie, Y., Yu, Y., & Huang, T. (2025). Graph Neural Networks for Fault Diagnosis in Photovoltaic-Integrated Distribution Networks with Weak Features. *Sensors*, 25(18), 5691.

5. Mansouri, M., et al. (2026). Physics-Embedded Hybrid Neural ODE With Digital Twin Residuals for Fault Diagnosis in Grid-Connected Photovoltaic Systems. *IEEE Transactions*.

6. DPAL: A dual-threshold physics-informed automatic labeling framework with open-set recognition for robust fault diagnosis of grid-connected photovoltaic systems under label scarcity. (2026). *ScienceDirect*.

7. Othman, R. A., & Al-Yozbaky, O. S. A.-D. (2024). Assessment of the efficiency and performance of different PV system configurations under various fault conditions. *IJPEDS*.

8. Validasi data NASA POWER untuk radiasi matahari di wilayah tropis. *Garuda Kemdiktisaintek*.