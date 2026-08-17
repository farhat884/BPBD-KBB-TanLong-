"""
elevation_engine.py
====================

Modul ini menambahkan FAKTOR TOPOGRAFI (dataran tinggi / dataran rendah)
ke dalam perhitungan potensi risiko tanah longsor.

Alasan kenapa ini relevan buat longsor:
- Wilayah dataran tinggi / pegunungan umumnya punya lereng yang lebih
  curam, curah hujan orografis lebih tinggi, dan tanah yang lebih labil
  di daerah perbukitan -- sehingga potensi longsornya secara umum lebih
  tinggi dibanding dataran rendah yang relatif datar.
- Modul ini TIDAK menggantikan "Kelas_Risiko" yang sudah ada di data
  Excel (yang datanya berasal dari kajian resmi kebencanaan), tapi
  dipakai sebagai FAKTOR TAMBAHAN yang digabung jadi satu skor baru:
  "Skor_Risiko_Gabungan" & "Kategori_Risiko_Gabungan".

CARA KERJA:
1. Hitung titik tengah (centroid) tiap kecamatan dari file
   static/id3217_bandung_barat/32.17_kecamatan.geojson (tidak perlu
   library tambahan seperti shapely -- cukup rata-rata koordinat titik
   poligon, cukup akurat untuk kebutuhan klasifikasi ketinggian).
2. Query ketinggian (mdpl) tiap titik tengah itu ke Open-Elevation API
   (gratis, tanpa API key): https://api.open-elevation.com
3. Hasilnya di-cache ke data/elevasi_cache.json supaya tidak nge-hit
   API terus-terusan tiap kali server restart (ketinggian kan tidak
   berubah-ubah).
4. Kalau API-nya lagi tidak bisa diakses (mis. tidak ada internet,
   domain diblokir firewall/proxy, dsb), modul ini otomatis jatuh ke
   FALLBACK_ELEVASI -- tabel referensi ketinggian per kecamatan yang
   sudah diketahui dari sumber publik (BPS Kab. Bandung Barat, kajian
   RTRW, dsb) supaya sistem tetap jalan dan tidak error.

CATATAN UNTUK YANG DEPLOY:
- Kalau di server deploy-mu (misalnya Vercel) domain
  api.open-elevation.com diblokir juga oleh firewall/proxy, sistem
  akan otomatis pakai FALLBACK_ELEVASI di bawah -- tidak akan bikin
  aplikasi down. Kalau mau data real-time yang lebih presisi per desa
  (bukan cuma per kecamatan), tinggal perluas OPEN_ELEVATION_URL untuk
  dipanggil per titik desa juga (sudah disiapkan fungsi generiknya).
"""

import os
import json
import time
import requests

OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
REQUEST_TIMEOUT = 6  # detik, jangan lama-lama biar server tidak macet

# ================================================================
# TABEL FALLBACK -- dipakai kalau API elevasi tidak bisa diakses.
# Sumber: kajian RTRW / gambaran umum wilayah Kabupaten Bandung
# Barat (BPS & studi topografi tiap kecamatan). Angka median/​
# perkiraan tengah dari rentang ketinggian tiap kecamatan (mdpl).
# Silakan dikoreksi kalau BPBD KBB punya data DEM/kontur yang lebih
# presisi.
# ================================================================
FALLBACK_ELEVASI = {
    "lembang": 1250,
    "cisarua": 1100,
    "parongpong": 1150,
    "cikalongwetan": 650,
    "cipeundeuy": 550,
    "cipatat": 500,
    "padalarang": 500,
    "ngamprah": 750,
    "batujajar": 650,
    "cihampelas": 500,
    "cililin": 550,
    "cipongkor": 600,
    "sindangkerta": 600,
    "gununghalu": 650,
    "rongga": 700,
    "saguling": 650,
}

DEFAULT_ELEVASI = 600  # kalau kecamatan sama sekali tidak dikenali


def klasifikasi_topografi(elevasi_m):
    """
    Klasifikasi sederhana dataran berdasar ketinggian (mdpl):
      < 500 m         : Dataran Rendah
      500 - 999 m      : Dataran Sedang / Perbukitan
      >= 1000 m        : Dataran Tinggi / Pegunungan
    """
    if elevasi_m is None:
        elevasi_m = DEFAULT_ELEVASI

    if elevasi_m >= 1000:
        return "Dataran Tinggi / Pegunungan"
    elif elevasi_m >= 500:
        return "Dataran Sedang / Perbukitan"
    else:
        return "Dataran Rendah"


def skor_topografi(kategori):
    """
    Bobot tambahan yang dipakai untuk menaikkan skor risiko gabungan.
    Semakin tinggi dataran -> semakin besar bobot tambahannya.
    """
    return {
        "Dataran Rendah": 0.0,
        "Dataran Sedang / Perbukitan": 0.5,
        "Dataran Tinggi / Pegunungan": 1.0,
    }.get(kategori, 0.0)


def _hitung_centroid_multipolygon(geometry):
    """
    Menghitung titik tengah (rata-rata) semua koordinat dari geometry
    tipe Polygon / MultiPolygon. Ini bukan centroid geometris yang
    presisi secara matematis, tapi cukup akurat untuk kebutuhan ambil
    1 titik wakil ketinggian per kecamatan.
    """
    lons, lats = [], []

    coords = geometry.get("coordinates", [])
    gtype = geometry.get("type", "")

    def _walk(c, depth):
        if depth == 0:
            lon, lat = c[0], c[1]
            lons.append(lon)
            lats.append(lat)
        else:
            for item in c:
                _walk(item, depth - 1)

    # Polygon -> depth koordinat titik ada di level ke-2 (ring -> titik)
    # MultiPolygon -> depth ada di level ke-3 (polygon -> ring -> titik)
    depth = 2 if gtype == "Polygon" else 3

    try:
        _walk(coords, depth)
    except Exception:
        pass

    if not lons or not lats:
        return None, None

    return (sum(lat for lat in lats) / len(lats),
            sum(lon for lon in lons) / len(lons))


def _load_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache_path, data):
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _query_elevation_batch(points):
    """
    points: list of (lat, lon)
    return: list of elevasi (meter) sesuai urutan points, atau None
            kalau request gagal semua.
    """
    if not points:
        return []

    payload = {
        "locations": [
            {"latitude": lat, "longitude": lon} for lat, lon in points
        ]
    }

    try:
        resp = requests.post(
            OPEN_ELEVATION_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        hasil = resp.json().get("results", [])
        return [h.get("elevation") for h in hasil]
    except Exception as e:
        print(f"⚠️  Gagal ambil data elevasi dari Open-Elevation API: {e}")
        return None


def hitung_elevasi_kecamatan(app_root_path, force_refresh=False):
    """
    Mengembalikan dict:
        {
          <nama_kecamatan_clean>: {
              "elevasi_m": <int>,
              "kategori_topografi": <str>,
              "sumber": "api" | "fallback" | "cache",
          },
          ...
        }
    """
    from ml_engine import clean_name  # import lokal, hindari circular import

    cache_path = os.path.join(app_root_path, "data", "elevasi_cache.json")

    if not force_refresh:
        cached = _load_cache(cache_path)
        if cached:
            return cached

    kec_file = os.path.join(
        app_root_path, "static", "id3217_bandung_barat", "32.17_kecamatan.geojson"
    )

    hasil = {}

    if not os.path.exists(kec_file):
        # Tidak ada file geojson kecamatan -> langsung pakai fallback penuh
        for nama, elevasi in FALLBACK_ELEVASI.items():
            kategori = klasifikasi_topografi(elevasi)
            hasil[nama] = {
                "elevasi_m": elevasi,
                "kategori_topografi": kategori,
                "sumber": "fallback",
            }
        _save_cache(cache_path, hasil)
        return hasil

    with open(kec_file, "r", encoding="utf-8") as f:
        kec_geojson = json.load(f)

    nama_list = []
    titik_list = []

    for feature in kec_geojson.get("features", []):
        props = feature.get("properties", {})
        nama_kec = str(props.get("nm_kecamatan", "")).strip()
        if not nama_kec:
            continue

        lat, lon = _hitung_centroid_multipolygon(feature.get("geometry", {}))
        if lat is None:
            continue

        nama_list.append(clean_name(nama_kec))
        titik_list.append((lat, lon))

    elevasi_api = _query_elevation_batch(titik_list)

    for i, nama in enumerate(nama_list):
        elevasi = None
        sumber = "fallback"

        if elevasi_api is not None and i < len(elevasi_api) and elevasi_api[i] is not None:
            elevasi = round(elevasi_api[i])
            sumber = "api"
        else:
            elevasi = FALLBACK_ELEVASI.get(nama, DEFAULT_ELEVASI)
            sumber = "fallback"

        kategori = klasifikasi_topografi(elevasi)

        hasil[nama] = {
            "elevasi_m": elevasi,
            "kategori_topografi": kategori,
            "sumber": sumber,
        }

    # Kecamatan di fallback yang mungkin tidak ada di geojson (jaga-jaga)
    for nama, elevasi in FALLBACK_ELEVASI.items():
        if nama not in hasil:
            kategori = klasifikasi_topografi(elevasi)
            hasil[nama] = {
                "elevasi_m": elevasi,
                "kategori_topografi": kategori,
                "sumber": "fallback",
            }

    _save_cache(cache_path, hasil)
    return hasil
