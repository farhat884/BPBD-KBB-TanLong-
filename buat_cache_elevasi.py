"""
buat_cache_elevasi.py
=====================
Bangun ulang data/elevasi_cache.json dari API Open-Elevation (data
ketinggian asli per kecamatan).

KENAPA ADA FILE INI
-------------------
Tanpa file cache, SETIAP kali server start (di hosting serverless = tiap
cold start) aplikasi memanggil api.open-elevation.com dengan batas tunggu
6 detik -- membuat halaman pertama lama terbuka. File cache yang ikut
di-deploy menghilangkan panggilan itu sama sekali.

Cache bawaan yang disertakan dibuat dari tabel referensi FALLBACK_ELEVASI
di elevation_engine.py (nilai yang sama yang dipakai aplikasi setiap API
gagal). Kalau mau memakai data API asli:

    python buat_cache_elevasi.py

jalankan di KOMPUTER YANG PUNYA INTERNET (bukan di server deploy), lalu
commit/deploy file data/elevasi_cache.json yang dihasilkan.
"""

import os

from elevation_engine import hitung_elevasi_kecamatan

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    hasil = hitung_elevasi_kecamatan(root, force_refresh=True)

    per_sumber = {}
    for info in hasil.values():
        per_sumber[info["sumber"]] = per_sumber.get(info["sumber"], 0) + 1

    print(f"Cache ditulis: {os.path.join(root, 'data', 'elevasi_cache.json')}")
    print(f"Sumber data per kecamatan: {per_sumber}")
    if per_sumber.get("api", 0) == 0:
        print(
            "⚠️  Tidak ada data dari API (semua fallback). Kemungkinan tidak ada "
            "internet / API sedang diblokir. Cache yang tertulis = tabel fallback."
        )
