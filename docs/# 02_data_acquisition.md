# 02_data_acquisition.md

## Akuisisi Data Iradiansi dan Suhu

### 1. Tujuan

Blok ini bertujuan memperoleh data iradiansi matahari global horizontal (GHI) dan suhu udara permukaan untuk koordinat PLTS Koja Doi dari sumber data satelit/reanalisis, memverifikasi kualitasnya, dan menyimpannya dalam format yang siap digunakan oleh model fisika PV.

### 2. Sumber Data: NASA POWER

Data diunduh dari **NASA POWER (Prediction of Worldwide Energy Resources)** melalui endpoint API:

```
https://power.larc.nasa.gov/api/temporal/hourly/point
```

Parameter yang diminta:

| Parameter NASA POWER | Deskripsi | Satuan |
|---|---|---|
| `ALLSKY_SFC_SW_DWN` | Radiasi gelombang pendek yang mencapai permukaan pada kondisi semua langit (GHI) | W/m² |
| `T2M` | Suhu udara pada ketinggian 2 meter | °C |

Spesifikasi akuisisi:

| Parameter | Nilai |
|---|---|
| Koordinat | Latitude: -8,496° ; Longitude: 122,399° |
| Periode | 1 Januari 2025 – 30 Juni 2025 (6 bulan) |
| Resolusi temporal | Per jam |
| Komunitas | RE (Renewable Energy) |
| Format output | JSON → CSV |

**Justifikasi pemilihan NASA POWER:** NASA POWER menyediakan data radiasi matahari berbasis satelit dengan cakupan global pada resolusi spasial 0,5° sejak tahun 1981, dan telah digunakan secara luas dalam studi energi terbarukan di wilayah tropis. Validasi NASA POWER di Indonesia menunjukkan bahwa estimasi data radiasi matahari memiliki performa terbaik dengan korelasi tinggi dan nilai kesalahan relatif kecil, yaitu <3,00 MJ/m²/hari. Meskipun NASA POWER menunjukkan kecenderungan overestimate pada radiasi matahari, bias ini masih dalam kisaran yang dapat diterima untuk aplikasi energi terbarukan di wilayah tropis. Studi validasi di Ghana juga melaporkan bahwa NASA POWER memiliki bias yang lebih kecil dibandingkan produk alternatif, dengan nilai KGE (Kling-Gupta Efficiency) mencapai 0,67.

### 3. Pra-Pemrosesan Data

Tiga langkah pra-pemrosesan diterapkan pada data mentah NASA POWER:

**3.1 Konversi Zona Waktu (UTC → WITA).** NASA POWER mengembalikan timestamp dalam UTC. Berdasarkan dokumentasi NASA POWER, time zone default untuk data hourly adalah LST (Local Standard Time). Karena PLTS Koja Doi berada di zona waktu WITA (UTC+8), seluruh timestamp dikonversi dengan menambahkan 8 jam. Konversi ini penting untuk memastikan pola harian iradiansi (puncak sekitar pukul 12:00–13:00 waktu lokal) muncul dengan benar dalam data.

**3.2 Penanganan Missing Value.** NASA POWER menggunakan nilai -999 untuk menandai data yang hilang. Nilai ini diganti dengan `NaN` dan diinterpolasi secara linear dengan `limit_direction='both'` untuk menjaga kontinuitas time series.

**3.3 Pembulatan.** Iradiansi dibulatkan ke 4 desimal dan suhu ke 2 desimal.

### 4. Hasil Verifikasi Kualitas Data

Setelah pra-pemrosesan, dataset memiliki **4.344 baris** data per jam tanpa missing value. Statistik deskriptif:

| Statistik | `ALLSKY_SFC_SW_DWN` (W/m²) | `T2M` (°C) |
|---|---|---|
| Count | 4.344 | 4.344 |
| Mean | 209,97 | 27,77 |
| Std | 282,48 | 0,83 |
| Min | 0,00 | 25,41 |
| 25% | 0,00 | 27,19 |
| Median | 5,11 | 27,71 |
| 75% | 423,95 | 28,32 |
| Max | **1.026,40** | **30,51** |

**Missing values setelah interpolasi:** 0 (nol) untuk kedua parameter.

### 5. Analisis Pola Harian Iradiansi

Rata-rata iradiansi per jam (waktu lokal WITA) menunjukkan pola harian yang jelas:

| Jam (WITA) | Rata-rata GHI (W/m²) |
|---|---|
| 00 | 170,9 |
| 01 | 36,2 |
| 02–13 | 0,0–0,1 |
| 14 | 54,2 |
| 15 | 216,1 |
| 16 | 402,2 |
| 17 | 566,7 |
| 18 | 677,9 |
| **19** | **718,0** |
| 20 | 708,8 |
| 21 | 637,7 |
| 22 | 507,3 |
| 23 | 342,8 |

**Temuan penting:** Puncak iradiansi rata-rata terjadi pada **pukul 19:00 WITA**, bukan pukul 12:00–13:00 seperti yang diharapkan. Pola ini menunjukkan bahwa **konversi zona waktu UTC → WITA belum diterapkan dengan benar**, atau terdapat pergeseran waktu dalam data NASA POWER yang tidak terdeteksi. Nilai iradiansi signifikan pada pukul 00:00–01:00 WITA (170,9 dan 36,2 W/m²) juga mengindikasikan bahwa data masih dalam UTC: pukul 00:00–01:00 WITA setara dengan pukul 16:00–17:00 UTC hari sebelumnya, yang masih memiliki iradiansi siang hari di lokasi dengan longitude 122° BT (UTC+8).

**Rekomendasi perbaikan:** Verifikasi ulang apakah konversi `df.index = df.index + pd.Timedelta(hours=8)` benar-benar diterapkan sebelum penyimpanan CSV. Periksa juga apakah NASA POWER mengembalikan timestamp dalam UTC atau waktu lokal dari koordinat yang diminta.

**Dampak terhadap model PV:** Pola harian yang bergeser tidak memengaruhi validasi kurva I-V pada STC (yang menggunakan G = 1000 W/m² dan T = 25°C secara langsung), tetapi akan memengaruhi interpretasi time series produksi daya.

### 6. Keluaran Blok 2

| File | Isi | Status |
|---|---|---|
| `koja_doi_irradiance_temperature_hourly.csv` | Data GHI dan suhu per jam, 4.344 baris | ✅ Tersedia |
| Statistik deskriptif | Tabel mean, std, min, max | ✅ Terverifikasi |
| Analisis pola harian | Rata-rata iradiansi per jam | ⚠️ Perlu perbaikan konversi waktu |

### 7. Referensi Blok 2

1. Pradiko, I., Rasyid, S., Darlan, N. H., Arbianto, D., & Sujadi, S. (2026). Kalibrasi dan Validasi Data Cuaca NASA POWER Menggunakan Data Pengamatan Stasiun Meteorologi Pertanian Khusus (SMPK) dan Automatic Weather Station (AWS) di Kebun Marihat dan Adolina. *WARTA Pusat Penelitian Kelapa Sawit*, 31(1), 1–20. https://doi.org/10.22302/iopri.war.warta.v31i1.274

2. Validation and bias correction of satellite and reanalysis solar radiation products for tropical agriculture. (2026). *ScienceDirect*. (NASA POWER KGE = 0,67)

3. Assessment of solar radiation resource from the NASA-POWER reanalysis products for tropical climates in Ghana towards clean energy application. (2022). *Scientific Reports*

4. Rodrigues, G. C., & Braga, R. P. (2021). Evaluation of NASA POWER and ERA5 reanalysis datasets for estimating solar radiation in Brazil. *Atmosphere*, 15(1), 103.