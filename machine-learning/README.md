# Klasifikasi Kelulusan Mahasiswa UNPAM

Project ini bertujuan memprediksi **Status_Kelulusan** (Lulus/Tidak Lulus) mahasiswa UNPAM menggunakan beberapa algoritma klasifikasi.

## Struktur
```
kelulusan_unpam_project_final/
├── data/kelulusan_unpam.csv
├── src/model_klasifikasi_unpam.py
├── notebooks/klasifikasi_unpam.ipynb
├── report/laporan_singkat.pdf
├── requirements.txt
└── README.md
```

## Cara menjalankan
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Jalankan script utama:
```
python src/model_klasifikasi_unpam.py --data data/kelulusan_unpam.csv --outdir outputs
```

3. Notebook Jupyter tersedia di `notebooks/klasifikasi_unpam.ipynb`.

## Penulis
- Maria Nagita Tria Vanessa (NIM: 231011400228) - 05 TPLE 005
