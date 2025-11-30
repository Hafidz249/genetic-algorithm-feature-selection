# genetic-algorithm-feature-selection
Implementasi Algoritma Genetika untuk Feature Selection pada dataset Breast Cancer menggunakan Python. Tugas Mata Kuliah Pembelajaran Mesin (Mechine Learning).
# Genetic Algorithm for Feature Selection 🧬

Repositori ini berisi implementasi **Algoritma Genetika (Genetic Algorithm)** untuk melakukan seleksi fitur (*Feature Selection*) pada dataset *Machine Learning*. [cite_start]Proyek ini disusun untuk memenuhi tugas mata kuliah (Bab 13: Genetic Algorithm)[cite: 1, 496].

## 📋 Deskripsi Proyek

[cite_start]Algoritma Genetika (GA) adalah metode optimasi dan pencarian yang terinspirasi oleh proses seleksi alam dan evolusi biologis[cite: 7, 8, 14]. Dalam proyek ini, kami menggunakan GA untuk memilih kombinasi fitur terbaik dari dataset agar model klasifikasi (Decision Tree) dapat mencapai akurasi maksimal.

[cite_start]Representasi individu menggunakan bilangan biner (1 untuk fitur terpilih, 0 untuk fitur dibuang) sesuai dengan teori representasi kromosom[cite: 30, 31, 114].

## ⚙️ Alur Algoritma

[cite_start]Implementasi kode mengikuti tahapan standar Algoritma Genetika[cite: 108]:

1.  [cite_start]**Inisialisasi Populasi**: Membangkitkan populasi awal berupa kromosom acak (biner)[cite: 111, 114].
2.  [cite_start]**Perhitungan Fitness**: Mengukur kualitas individu berdasarkan akurasi model pada data uji[cite: 69, 103].
3.  [cite_start]**Seleksi (Selection)**: Menggunakan metode **Roulette Wheel Selection** untuk memilih induk terbaik[cite: 177, 195].
4.  [cite_start]**Pindah Silang (Crossover)**: Menggunakan metode **One-Point Crossover** dengan probabilitas tertentu untuk menghasilkan keturunan baru[cite: 199, 221].
5.  [cite_start]**Mutasi (Mutation)**: Menggunakan metode **Bit Inversion/Flip** (mengubah 0 jadi 1 atau sebaliknya) untuk menjaga keragaman genetik[cite: 283, 304].
6.  [cite_start]**Elitisme**: Menyimpan individu terbaik (Elite) ke generasi berikutnya agar solusi terbaik tidak hilang[cite: 333].

## 🛠️ Teknologi yang Digunakan

* **Bahasa**: Python
* **Library**:
    * `pandas` (Manipulasi data)
    * `numpy` (Operasi numerik)
    * `scikit-learn` (Model Machine Learning & Evaluasi)
    * `matplotlib` (Visualisasi hasil evolusi)

## 📂 Struktur File
