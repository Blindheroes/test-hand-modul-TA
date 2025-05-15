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

Anda dapat menjalankan semua pengujian atau pengujian tertentu saja menggunakan skrip `run_tests.py`.

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

1. Tekan tombol '1', '2', atau '3' untuk memilih jarak pengujian:

   - '1' = Jarak Dekat (1 meter)
   - '2' = Jarak Optimal (2 meter)
   - '3' = Jarak Jauh (3 meter)

2. Setelah memilih jarak, pilih gerakan yang akan diuji:

   - '1' = Gerakan Pointing (jari telunjuk)
   - '2' = Gerakan Selecting (telunjuk dan kelingking)
   - '0' = Kembali ke pemilihan jarak

3. Setelah memilih gerakan, tunggu hitungan mundur 5 detik untuk bersiap

4. Lakukan gerakan yang dipilih selama 20 detik

   - Sistem akan menampilkan status deteksi gerakan secara real-time
   - Sistem mencatat setiap frame dan status deteksi

5. Setelah pengujian selesai, Anda akan kembali ke menu pemilihan gerakan
   - Anda dapat memilih gerakan lain pada jarak yang sama
   - Atau kembali ke pemilihan jarak untuk menguji pada jarak berbeda

Pengujian akan otomatis mencatat:

- Timestamp setiap frame
- Nomor frame
- Jarak yang digunakan
- Gerakan yang sedang diuji
- Status deteksi (1 jika terdeteksi, 0 jika tidak terdeteksi)

### Pengujian Pencahayaan (Lighting Test)

1. Tekan tombol '1', '2', atau '3' untuk memilih kondisi pencahayaan:

   - '1' = Pencahayaan Rendah (~50 lux)
   - '2' = Pencahayaan Sedang (~300 lux)
   - '3' = Pencahayaan Tinggi (~600 lux)

2. Setelah memilih pencahayaan, pilih gerakan yang akan diuji:

   - '1' = Gerakan Pointing (jari telunjuk)
   - '2' = Gerakan Selecting (telunjuk dan kelingking)
   - '0' = Kembali ke pemilihan pencahayaan

3. Setelah memilih gerakan, tunggu hitungan mundur 5 detik untuk bersiap

4. Lakukan gerakan yang dipilih selama 20 detik

   - Sistem akan menampilkan status deteksi gerakan dan tingkat kecerahan frame secara real-time
   - Sistem mencatat setiap frame dan status deteksi

5. Setelah pengujian selesai, Anda akan kembali ke menu pemilihan gerakan
   - Anda dapat memilih gerakan lain pada kondisi pencahayaan yang sama
   - Atau kembali ke pemilihan pencahayaan untuk menguji pada pencahayaan berbeda

Pengujian akan otomatis mencatat:

- Timestamp setiap frame
- Nomor frame
- Kondisi pencahayaan
- Gerakan yang sedang diuji
- Tingkat kecerahan frame (brightness)
- Status deteksi (1 jika terdeteksi, 0 jika tidak terdeteksi)

### Pengujian Adaptasi Threshold (Threshold Test)

1. Tekan tombol '1', '2', atau '3' untuk memilih jarak pengujian:

   - '1' = Jarak Dekat (1 meter)
   - '2' = Jarak Optimal (2 meter)
   - '3' = Jarak Jauh (3 meter)

2. Setelah memilih jarak, tunggu hitungan mundur 5 detik untuk bersiap

3. Pengujian akan dibagi menjadi dua fase:

   - Fase pertama (10 detik): Dengan mode adaptasi threshold (threshold disesuaikan dengan jarak)
   - Fase kedua (10 detik): Dengan mode threshold tetap (threshold tidak menyesuaikan jarak)

4. Selama pengujian:

   - Lakukan gerakan pointing dan selecting secara bergantian
   - Amati bagaimana sistem mendeteksi gerakan pada jarak yang dipilih
   - Perhatikan nilai threshold yang ditampilkan di layar

5. Setelah pengujian selesai, Anda kembali ke menu pemilihan jarak
   - Anda dapat memilih jarak lain untuk pengujian berikutnya
   - Atau keluar dengan menekan ESC

Pengujian akan otomatis mencatat:

- Timestamp setiap frame
- Nomor frame
- Mode threshold (adaptif atau tetap)
- Jarak pengujian (dekat, optimal, atau jauh)
- Nilai threshold yang digunakan
- Gerakan yang terdeteksi (atau tidak terdeteksi)
- Status deteksi (1 jika terdeteksi, 0 jika tidak terdeteksi)

### Pengujian Pemfilteran Temporal (Temporal Test)

1. Tekan tombol '1' atau '2' untuk memilih mode pemfilteran:

   - '1' = Mode Dengan Pemfilteran (with_filtering)
   - '2' = Mode Tanpa Pemfilteran (without_filtering)

2. Setelah memilih mode, tunggu hitungan mundur 5 detik untuk bersiap

3. Pengujian akan berlangsung selama 20 detik:

   - Fase pertama (20 detik): Pengujian dengan mode pemfilteran yang dipilih
   - Setelah selesai, Anda dapat memilih mode lain atau keluar

4. Selama pengujian:

   - Lakukan gerakan pointing dan selecting dengan cepat bergantian
   - Amati perbedaan antara deteksi mentah (raw) dan terfilter
   - Perhatikan bagaimana stabilitas deteksi berbeda antara mode dengan dan tanpa pemfilteran

5. Setelah pengujian selesai, Anda kembali ke menu pemilihan mode
   - Anda dapat memilih mode lain untuk pengujian berikutnya
   - Atau keluar dengan menekan ESC

Pengujian akan otomatis mencatat:

- Timestamp setiap frame
- Mode pemfilteran (dengan pemfilteran atau tanpa pemfilteran)
- Gerakan yang terdeteksi (pointing atau selecting)
- Tipe deteksi (mentah atau terfilter)
- Status deteksi (1 jika terdeteksi, 0 jika tidak terdeteksi)
