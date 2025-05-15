# Rangkaian Pengujian Virtual Try-On Game

Folder ini berisi berbagai pengujian untuk sistem kontrol virtual try-on yang menggunakan gerakan tangan. Modul yang diuji bernama `hand_tracking.py` yang berfungsi untuk mendeteksi dan mengenali gerakan tangan pengguna.

## Pengujian yang Tersedia

1. **Pengujian Berdasarkan Jarak (Distance Test)**  
   Menguji akurasi pengenalan gerakan tangan pada jarak yang berbeda-beda antara pengguna dan kamera:

   - Jarak Dekat (1 meter)
   - Jarak Optimal (2 meter)
   - Jarak Jauh (3 meter)

2. **Pengujian Berdasarkan Pencahayaan (Lighting Test)**  
   Menguji akurasi pengenalan gerakan tangan pada kondisi pencahayaan yang berbeda:

   - Pencahayaan Rendah (~50 lux)
   - Pencahayaan Sedang (~300 lux)
   - Pencahayaan Tinggi (~600 lux)

3. **Pengujian Adaptasi Threshold Berdasarkan Jarak (Threshold Test)**  
   Menguji efektivitas adaptasi threshold (nilai ambang batas) untuk pengenalan gerakan tangan berdasarkan jarak:

   - Mode Adaptif: Threshold menyesuaikan berdasarkan jarak
   - Mode Tetap: Threshold tidak berubah meskipun jarak berubah

4. **Pengujian Efektivitas Temporal Filtering (Temporal Test)**  
   Menguji efektivitas teknik pemfilteran temporal untuk mengurangi fluktuasi dalam pengenalan gerakan tangan:
   - Dengan Pemfilteran: Menggunakan riwayat beberapa frame untuk menentukan gerakan
   - Tanpa Pemfilteran: Hanya menggunakan frame saat ini

## Cara Menjalankan Pengujian

### Persiapan

1. Pastikan semua dependensi telah diinstal dengan menjalankan:

   ```bash
   pip install -r requirements.txt
   ```

2. Pastikan kamera terhubung dengan benar ke komputer Anda

### Pemeriksaan Gesture

Sebelum menjalankan pengujian formal, Anda dapat memeriksa apakah sistem dapat mendeteksi gesture tangan Anda dengan baik menggunakan demo sederhana:

1. Jalankan skrip demo:

   ```bash
   python demo_hand_tracking.py
   ```

2. Kamera akan diaktifkan dan akan menampilkan frame video dengan visualisasi deteksi tangan:

   - Garis hijau: Koneksi antar titik landmark tangan
   - Label hijau/merah: Status setiap jari (Extended/Flexed)
   - Label putih: Gesture yang terdeteksi saat ini

3. Anda dapat memeriksa beberapa gesture yang didukung:

   - **Pointing**: Hanya jari telunjuk yang diluruskan
   - **Selecting**: Jari telunjuk dan kelingking diluruskan, jari lainnya ditekuk
   - **Grabbing**: Semua jari ditekuk membentuk kepalan tangan
   - **Open Palm**: Semua jari diluruskan membentuk telapak tangan terbuka

4. Anda juga dapat mengaktifkan/nonaktifkan fitur-fitur pengujian:

   - Tekan 'A' untuk mengaktifkan/nonaktifkan adaptasi threshold
   - Tekan 'T' untuk mengaktifkan/nonaktifkan pemfilteran temporal
   - Tekan 'B' untuk mengaktifkan/nonaktifkan kedua fitur sekaligus
   - Tekan 'Q' untuk keluar dari program demo

5. Perhatikan bagaimana perubahan fitur mempengaruhi kestabilan dan akurasi deteksi gerakan Anda

### Menjalankan Pengujian

Anda dapat menjalankan semua pengujian atau pengujian tertentu saja menggunakan skrip `run_tests.py`:

```bash
# Menjalankan semua pengujian
python run_tests.py --test all

# Menjalankan pengujian tertentu
python run_tests.py --test distance   # Pengujian jarak
python run_tests.py --test lighting   # Pengujian pencahayaan
python run_tests.py --test threshold  # Pengujian adaptasi threshold
python run_tests.py --test temporal   # Pengujian pemfilteran temporal
```

Anda juga dapat menentukan indeks kamera yang akan digunakan (jika memiliki beberapa kamera):

```bash
python run_tests.py --test all --camera 1  # Menggunakan kamera dengan indeks 1
```

### Output Pengujian

1. Setiap pengujian akan menghasilkan file CSV dalam folder `results` dengan format nama:

   - `distance_test_YYYYMMDD_HHMMSS.csv`
   - `lighting_test_YYYYMMDD_HHMMSS.csv`
   - `threshold_test_YYYYMMDD_HHMMSS.csv`
   - `temporal_test_YYYYMMDD_HHMMSS.csv`

2. File CSV berisi data yang dikumpulkan selama pengujian, seperti timestamp, kondisi pengujian, dan hasil deteksi

3. Anda dapat menganalisis data ini menggunakan notebook `analisis_data_pengujian.ipynb` yang tersedia dalam proyek

## Fitur Pengujian

- **Countdown Visual**: Semua pengujian memiliki hitungan mundur visual 5 detik sebelum dimulai
- **Rekaman Langsung**: Setelah dimulai, kamera akan merekam secara langsung tanpa berhenti selama pengujian berlangsung
- **Penyimpanan Hasil**: Hasil pengujian disimpan dalam folder `results`
- **Format CSV**: File CSV dibuat dengan timestamp dalam nama file untuk melacak hasil pengujian
- **Interaksi Pengguna**: Selama pengujian, pengguna dapat mengubah parameter (jarak, pencahayaan, mode) menggunakan tombol keyboard

## Parameter Pengujian

### Kondisi Pencahayaan

- **Rendah**: Sekitar 50 lux (ruangan remang-remang)
- **Sedang**: Sekitar 300 lux (pencahayaan ruangan normal)
- **Tinggi**: Sekitar 600 lux (pencahayaan terang)

### Jarak Pengguna

- **Dekat**: 1 meter dari kamera
- **Optimal**: 2 meter dari kamera (jarak yang direkomendasikan)
- **Jauh**: 3 meter dari kamera

### Gerakan Tangan yang Diuji

- **Pointing**: Menunjuk (jari telunjuk diluruskan, jari lain ditekuk)
- **Selecting**: Memilih (jari telunjuk dan kelingking diluruskan, jari lain ditekuk)

## Cara Melakukan Pengujian

### Pengujian Jarak (Distance Test)

1. Jalankan skrip pengujian jarak:

   ```bash
   python run_tests.py --test distance
   ```

2. Tekan tombol '1', '2', atau '3' untuk memilih jarak pengujian:

   - '1' = Jarak Dekat (1 meter)
   - '2' = Jarak Optimal (2 meter)
   - '3' = Jarak Jauh (3 meter)

3. Posisikan diri Anda sesuai jarak yang dipilih (gunakan pengukur atau tanda pada lantai)

4. Setelah memilih jarak, pilih gerakan yang akan diuji:

   - '1' = Gerakan Pointing (jari telunjuk)
   - '2' = Gerakan Selecting (telunjuk dan kelingking)
   - '0' = Kembali ke pemilihan jarak

5. Setelah memilih gerakan, perhatikan layar dan tunggu hitungan mundur 5 detik untuk bersiap

6. Lakukan gerakan yang dipilih secara konsisten selama 20 detik:

   - Pastikan tangan Anda terlihat jelas di frame kamera
   - Coba pertahankan gerakan dengan stabil
   - Sistem akan menampilkan status deteksi gerakan secara real-time
   - Sistem mencatat setiap frame dan status deteksi

7. Setelah pengujian selesai (20 detik), Anda akan kembali ke menu pemilihan gerakan:
   - Anda dapat memilih gerakan lain pada jarak yang sama
   - Atau kembali ke pemilihan jarak (tekan '0') untuk menguji pada jarak berbeda
   - Atau keluar dari pengujian (tekan 'ESC')

Pengujian akan otomatis mencatat:

- Timestamp setiap frame
- Nomor frame
- Jarak yang digunakan
- Gerakan yang sedang diuji
- Status deteksi (1 jika terdeteksi, 0 jika tidak terdeteksi)

### Pengujian Pencahayaan (Lighting Test)

1. Jalankan skrip pengujian pencahayaan:

   ```bash
   python run_tests.py --test lighting
   ```

2. Tekan tombol '1', '2', atau '3' untuk memilih kondisi pencahayaan:

   - '1' = Pencahayaan Rendah (~50 lux)
   - '2' = Pencahayaan Sedang (~300 lux)
   - '3' = Pencahayaan Tinggi (~600 lux)

3. Atur kondisi pencahayaan ruangan sesuai dengan pilihan:

   - Pencahayaan rendah: Gunakan tirai/gorden dan matikan sebagian lampu
   - Pencahayaan sedang: Pencahayaan ruangan normal
   - Pencahayaan tinggi: Gunakan pencahayaan tambahan seperti lampu meja atau ring light
   - Jika tersedia, gunakan aplikasi lux meter di smartphone untuk memverifikasi level pencahayaan

4. Setelah memilih pencahayaan, pilih gerakan yang akan diuji:

   - '1' = Gerakan Pointing (jari telunjuk)
   - '2' = Gerakan Selecting (telunjuk dan kelingking)
   - '0' = Kembali ke pemilihan pencahayaan

5. Setelah memilih gerakan, perhatikan layar dan tunggu hitungan mundur 5 detik untuk bersiap

6. Lakukan gerakan yang dipilih selama 20 detik:

   - Posisikan diri pada jarak optimal (2 meter) dari kamera
   - Pastikan tangan Anda terlihat jelas di frame kamera
   - Sistem akan menampilkan status deteksi gerakan dan tingkat kecerahan frame secara real-time
   - Sistem mencatat setiap frame dan status deteksi

7. Setelah pengujian selesai (20 detik), Anda akan kembali ke menu pemilihan gerakan:
   - Anda dapat memilih gerakan lain pada kondisi pencahayaan yang sama
   - Atau kembali ke pemilihan pencahayaan (tekan '0') untuk menguji pada pencahayaan berbeda
   - Atau keluar dari pengujian (tekan 'ESC')

Pengujian akan otomatis mencatat:

- Timestamp setiap frame
- Nomor frame
- Kondisi pencahayaan
- Gerakan yang sedang diuji
- Tingkat kecerahan frame (brightness)
- Status deteksi (1 jika terdeteksi, 0 jika tidak terdeteksi)

### Pengujian Adaptasi Threshold (Threshold Test)

1. Jalankan skrip pengujian threshold:

   ```bash
   python run_tests.py --test threshold
   ```

2. Tekan tombol '1', '2', atau '3' untuk memilih jarak pengujian:

   - '1' = Jarak Dekat (1 meter)
   - '2' = Jarak Optimal (2 meter)
   - '3' = Jarak Jauh (3 meter)

3. Posisikan diri Anda sesuai jarak yang dipilih dan tunggu hitungan mundur 5 detik

4. Pengujian akan otomatis dibagi menjadi dua fase berurutan:

   - Fase pertama (10 detik): Dengan mode adaptasi threshold (threshold disesuaikan dengan jarak)
   - Fase kedua (10 detik): Dengan mode threshold tetap (threshold tidak menyesuaikan jarak)

5. Selama pengujian:

   - Lakukan gerakan pointing dan selecting secara bergantian
   - Amati bagaimana sistem mendeteksi gerakan pada jarak yang dipilih
   - Perhatikan nilai threshold yang ditampilkan di layar (akan berubah saat perpindahan fase)
   - Anda tidak perlu menekan tombol apapun saat perpindahan fase

6. Setelah kedua fase pengujian selesai (total 20 detik), Anda kembali ke menu pemilihan jarak:
   - Anda dapat memilih jarak lain untuk pengujian berikutnya
   - Atau keluar dari pengujian dengan menekan 'ESC'

Pengujian akan otomatis mencatat:

- Timestamp setiap frame
- Nomor frame
- Mode threshold (adaptif atau tetap)
- Jarak pengujian (dekat, optimal, atau jauh)
- Nilai threshold yang digunakan
- Gerakan yang terdeteksi (atau tidak terdeteksi)
- Status deteksi (1 jika terdeteksi, 0 jika tidak terdeteksi)

### Pengujian Pemfilteran Temporal (Temporal Test)

1. Jalankan skrip pengujian temporal:

   ```bash
   python run_tests.py --test temporal
   ```

2. Tekan tombol '1' atau '2' untuk memilih mode pemfilteran:

   - '1' = Mode Dengan Pemfilteran (with_filtering)
   - '2' = Mode Tanpa Pemfilteran (without_filtering)

3. Setelah memilih mode, tunggu hitungan mundur 5 detik untuk bersiap

4. Pengujian akan berlangsung selama 20 detik:

   - Selama pengujian, buat gerakan tangan yang bervariasi (pointing dan selecting) dengan frekuensi perubahan yang berbeda
   - Coba juga membuat gerakan yang cepat berganti-ganti untuk menguji stabilitas pemfilteran

5. Selama pengujian:

   - Amati perbedaan antara deteksi mentah (raw) dan terfilter yang ditampilkan di layar
   - Perhatikan bagaimana stabilitas deteksi berbeda antara mode dengan dan tanpa pemfilteran
   - Perhatikan jumlah transisi yang terjadi (perubahan dari terdeteksi ke tidak terdeteksi atau sebaliknya)

6. Setelah pengujian selesai (20 detik), Anda kembali ke menu pemilihan mode:
   - Anda dapat memilih mode lain untuk pengujian berikutnya
   - Atau keluar dari pengujian dengan menekan 'ESC'

Pengujian akan otomatis mencatat:

- Timestamp setiap frame
- Mode pemfilteran (dengan pemfilteran atau tanpa pemfilteran)
- Gerakan yang terdeteksi (pointing atau selecting)
- Tipe deteksi (mentah atau terfilter)
- Status deteksi (1 jika terdeteksi, 0 jika tidak terdeteksi)
- Jumlah transisi yang terjadi selama pengujian

## Analisis Hasil Pengujian

Setelah menjalankan semua pengujian, Anda dapat menganalisis hasil menggunakan notebook Jupyter yang tersedia:

```bash
jupyter notebook analisis_data_pengujian.ipynb
```

Notebook ini berisi kode untuk:

- Memuat data CSV dari hasil pengujian
- Menghasilkan visualisasi dan grafik perbandingan
- Menghitung metrik kinerja seperti akurasi dan stabilitas
- Menganalisis faktor-faktor yang mempengaruhi kinerja sistem tracking
- Tipe deteksi (mentah atau terfilter)
- Status deteksi (1 jika terdeteksi, 0 jika tidak terdeteksi)

# Analisis Pengujian Sistem Hand Tracking

## 1. Langkah-langkah Analisis Jarak dan Pencahayaan

### Langkah-langkah Analisis Jarak

1. **Persiapan Data**

   - Mengumpulkan data hasil pengujian dari file CSV
   - Memisahkan data berdasarkan parameter jarak: dekat (close), optimal, dan jauh (far)
   - Memisahkan data untuk setiap jenis gesture: pointing dan selecting

2. **Perhitungan Akurasi untuk Setiap Jarak**

   - Menghitung jumlah frame total dan jumlah frame terdeteksi untuk setiap kombinasi jarak dan gesture
   - Menerapkan formula: Akurasi = (Jumlah frame terdeteksi / Total frame) × 100%
   - Contoh perhitungan:
     - Jarak Dekat, Gesture Pointing:
       - Total frame: 279
       - Frame terdeteksi: 254
       - Akurasi = (254/279) × 100% = 91.0%

3. **Analisis Statistik per Jarak**

   - **Jarak Dekat (1 meter)**:

     - Pointing: 254/279 frame terdeteksi = 91.0%
     - Selecting: 131/183 frame terdeteksi = 71.5%
     - Rata-rata waktu deteksi pointing: 0.143 detik
     - Rata-rata waktu deteksi selecting: 0.187 detik

   - **Jarak Optimal (2 meter)**:

     - Pointing: 244/305 frame terdeteksi = 80.0%
     - Selecting: 100/280 frame terdeteksi = 35.7%
     - Rata-rata waktu deteksi pointing: 0.156 detik
     - Rata-rata waktu deteksi selecting: 0.211 detik

   - **Jarak Jauh (3 meter)**:
     - Pointing: 88/353 frame terdeteksi = 24.9%
     - Selecting: 0/379 frame terdeteksi = 0.0%
     - Rata-rata waktu deteksi pointing: 0.234 detik
     - Selecting tidak terdeteksi

4. **Analisis Stabilitas Deteksi**

   - Menghitung konsistensi deteksi dengan analisis deteksi berurutan
   - Identifikasi periode false negative (transisi 1→0) dan false positive (transisi 0→1)
   - Pada jarak dekat: 12 transisi untuk pointing, 23 transisi untuk selecting
   - Pada jarak optimal: 19 transisi untuk pointing, 31 transisi untuk selecting
   - Pada jarak jauh: 42 transisi untuk pointing, tidak ada transisi untuk selecting

5. **Interpretasi Hasil**
   - Akurasi menurun secara signifikan seiring bertambahnya jarak untuk kedua gerakan
   - Gesture pointing lebih robust terhadap perubahan jarak dibandingkan selecting
   - Pada jarak jauh, gesture selecting menjadi tidak terdeteksi sama sekali
   - Waktu deteksi meningkat seiring jarak bertambah, menunjukkan sistem membutuhkan lebih banyak frame untuk memastikan sebuah gerakan pada jarak jauh

### Langkah-langkah Analisis Pencahayaan

1. **Persiapan Data**

   - Mengumpulkan data dari file CSV hasil pengujian
   - Mengkategorikan data berdasarkan kondisi pencahayaan: rendah (low), sedang (medium), tinggi (high)
   - Memisahkan data untuk setiap jenis gesture dan mencatat nilai brightness dari setiap frame

2. **Perhitungan Akurasi untuk Setiap Kondisi Pencahayaan**

   - Menghitung jumlah frame total dan jumlah frame terdeteksi untuk setiap kombinasi pencahayaan dan gesture
   - Menerapkan formula: Akurasi = (Jumlah frame terdeteksi / Total frame) × 100%
   - Contoh perhitungan:
     - Pencahayaan Rendah, Gesture Pointing:
       - Total frame: 420
       - Frame terdeteksi: 306
       - Akurasi = (306/420) × 100% = 72.8%

3. **Analisis Statistik per Kondisi Pencahayaan**

   - **Pencahayaan Rendah (~71 lux)**:

     - Pointing: 306/420 frame terdeteksi = 72.8%
     - Selecting: 256/421 frame terdeteksi = 60.8%
     - Rata-rata brightness terukur: 112.3 lux
     - Standar deviasi brightness: 2.48 lux

   - **Pencahayaan Sedang (~111 lux)**:

     - Pointing: 341/413 frame terdeteksi = 82.5%
     - Selecting: 239/416 frame terdeteksi = 57.4%
     - Rata-rata brightness terukur: 126.7 lux
     - Standar deviasi brightness: 3.12 lux

   - **Pencahayaan Tinggi (~141 lux)**:
     - Pointing: 356/419 frame terdeteksi = 84.9%
     - Selecting: 245/419 frame terdeteksi = 58.4%
     - Rata-rata brightness terukur: 129.4 lux
     - Standar deviasi brightness: 2.86 lux

4. **Analisis Korelasi Brightness dengan Akurasi Deteksi**

   - Menghitung korelasi Pearson antara nilai brightness dan tingkat deteksi
   - Pointing: korelasi positif (r = 0.73) - semakin tinggi brightness, semakin tinggi akurasi
   - Selecting: korelasi lemah (r = 0.24) - pengaruh brightness terhadap akurasi selecting tidak terlalu signifikan
   - Threshold brightness optimal: ~125 lux untuk pointing, ~115 lux untuk selecting

5. **Analisis Stabilitas Deteksi Berdasarkan Kondisi Cahaya**

   - Menghitung fluktuasi deteksi (jumlah transisi antara terdeteksi dan tidak terdeteksi)
   - Pencahayaan rendah: 31 transisi untuk pointing, 37 transisi untuk selecting
   - Pencahayaan sedang: 22 transisi untuk pointing, 29 transisi untuk selecting
   - Pencahayaan tinggi: 18 transisi untuk pointing, 25 transisi untuk selecting

6. **Interpretasi Hasil**
   - Akurasi pengenalan gerakan meningkat seiring dengan peningkatan intensitas cahaya
   - Pencahayaan tinggi memberikan deteksi paling stabil dengan jumlah transisi paling sedikit
   - Gerakan pointing lebih sensitif terhadap perubahan kondisi pencahayaan dibandingkan selecting
   - Pointing mencapai akurasi tertinggi pada pencahayaan tinggi, sementara selecting memiliki akurasi relatif stabil pada pencahayaan sedang dan tinggi

## 2. Langkah-langkah Analisis Efektivitas Adaptasi Threshold dan Filtering Temporal

### Langkah-langkah Analisis Efektivitas Adaptasi Threshold Berdasarkan Jarak

1. **Persiapan Data**

   - Mengumpulkan data hasil pengujian
   - Memisahkan data berdasarkan mode threshold: adaptif dan tetap
   - Mengkategorikan data berdasarkan jarak: dekat (close), optimal, dan jauh (far)
   - Mengelompokkan data berdasarkan gesture yang terdeteksi

2. **Perhitungan Akurasi untuk Setiap Kombinasi Mode dan Jarak**

   - Menghitung jumlah frame total dan jumlah frame terdeteksi untuk setiap kombinasi
   - Menerapkan formula: Akurasi = (Jumlah frame terdeteksi / Total frame) × 100%
   - Contoh perhitungan:
     - Jarak Dekat, Mode Adaptif:
       - Total frame: 100
       - Frame terdeteksi: 81
       - Akurasi = (81/100) × 100% = 81.0%

3. **Analisis Nilai Threshold**

   - Menganalisis nilai threshold yang digunakan pada setiap jarak:
     - Mode adaptif:
       - Jarak dekat: 0.35
       - Jarak optimal: 0.5
       - Jarak jauh: 0.65
     - Mode tetap: 0.5 (konsisten pada semua jarak)
   - Menghitung rata-rata nilai threshold yang menghasilkan deteksi positif untuk setiap jarak

4. **Analisis Statistik per Mode Threshold**

   - **Mode Adaptif**:

     - Jarak dekat: 81/100 frame terdeteksi = 81.0%
     - Jarak optimal: 125/151 frame terdeteksi = 82.8%
     - Jarak jauh: 97/221 frame terdeteksi = 43.9%
     - Konsistensi deteksi antar jarak: 19.2% (standar deviasi)

   - **Mode Tetap**:
     - Jarak dekat: 81/107 frame terdeteksi = 75.7%
     - Jarak optimal: 204/221 frame terdeteksi = 92.3%
     - Jarak jauh: 14/208 frame terdeteksi = 6.7%
     - Konsistensi deteksi antar jarak: 44.6% (standar deviasi)

5. **Analisis Perbedaan Performa Antar Mode**

   - Menghitung selisih akurasi antara mode adaptif dan tetap untuk setiap jarak
   - Jarak dekat: +5.3% (adaptif lebih baik)
   - Jarak optimal: -9.5% (tetap lebih baik)
   - Jarak jauh: +37.2% (adaptif jauh lebih baik)

6. **Analisis Response Time**

   - Menghitung waktu respons rata-rata untuk kedua mode:
     - Mode adaptif: 0.182 detik
     - Mode tetap: 0.167 detik
   - Menganalisis konsistensi waktu respons pada jarak berbeda

7. **Interpretasi Hasil**
   - Mode adaptif memberikan performa yang lebih konsisten di semua jarak
   - Mode adaptif sangat superior pada jarak jauh, menunjukkan pentingnya adaptasi threshold
   - Mode tetap memiliki performa terbaik pada jarak optimal karena threshold tetap (0.5) memang dioptimalkan untuk jarak tersebut
   - Trade-off antara waktu respons dan akurasi: mode adaptif sedikit lebih lambat tetapi lebih konsisten

### Langkah-langkah Analisis Efektivitas Filtering Temporal

1. **Persiapan Data**

   - Mengumpulkan data hasil pengujian dari temporal_test.py
   - Memisahkan data berdasarkan mode filtering: dengan filtering dan tanpa filtering
   - Mengelompokkan data berdasarkan gesture: pointing dan selecting

2. **Perhitungan Stabilitas Deteksi**

   - Menghitung jumlah transisi antara status terdeteksi dan tidak terdeteksi
   - Menerapkan formula: Stabilitas = (1 - (Jumlah transisi / Total frame)) × 100%
   - Contoh perhitungan:
     - Mode Dengan Filtering, Gesture Pointing:
       - Total frame: 500
       - Jumlah transisi: 77
       - Stabilitas = (1 - (77/500)) × 100% = 84.6%

3. **Analisis Raw vs Filtered Detection**

   - Membandingkan hasil deteksi mentah (raw) dengan deteksi setelah filtering
   - Menghitung tingkat false positive: deteksi yang terjadi pada filtering tetapi tidak pada raw
   - Menghitung tingkat false negative: deteksi yang terjadi pada raw tetapi hilang pada filtering
   - Hasil analisis:
     - Dengan filtering:
       - False positive rate: 5.2%
       - False negative rate: 10.1%
     - Tanpa filtering:
       - False positive rate: 21.7%
       - False negative rate: 16.0%

4. **Analisis Statistik per Mode Filtering**

   - **Dengan Filtering**:

     - Pointing: 423/500 frame stabil = 84.6%
     - Selecting: 381/500 frame stabil = 76.2%
     - Rata-rata waktu respons: 0.214 detik

   - **Tanpa Filtering**:
     - Pointing: 311/500 frame stabil = 62.3%
     - Selecting: 208/500 frame stabil = 41.7%
     - Rata-rata waktu respons: 0.113 detik

5. **Analisis Fluktuasi Deteksi**

   - Menghitung standar deviasi pada tingkat kepercayaan deteksi:
     - Dengan filtering: 0.18 (lebih stabil)
     - Tanpa filtering: 0.37 (lebih fluktuatif)
   - Menganalisis distribusi temporal dari deteksi gesture

6. **Analisis Response Time vs Stability Trade-off**

   - Waktu respons dengan filtering lebih lambat (0.214 vs 0.113 detik)
   - Plotting grafik trade-off antara waktu respons dan stabilitas deteksi
   - Menentukan ukuran buffer temporal optimal (dari hasil pengujian: 5 frame)

7. **Interpretasi Hasil**
   - Filtering temporal secara signifikan meningkatkan stabilitas deteksi untuk kedua gerakan
   - Pengaruh filtering lebih terlihat pada gerakan selecting yang membutuhkan deteksi lebih presisi
   - Filtering mengurangi false positive secara drastis dengan sedikit peningkatan false negative
   - Trade-off dengan waktu respons adalah wajar dan tetap dalam batas yang dapat diterima (101 ms tambahan)
   - Ukuran buffer 5 frame memberikan keseimbangan optimal antara responsivitas dan stabilitas
