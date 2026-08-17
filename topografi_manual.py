"""
topografi_manual.py
====================

Fitur "Input Topografi Manual via Gambar" -- alternatif untuk
elevation_engine.py (yang mengandalkan Open-Elevation API + tabel
fallback statis per kecamatan).

LATAR BELAKANG:
- Open-Elevation itu sebenarnya API GRATIS (tanpa API key), tapi kalau
  server deploy (mis. Vercel) memblokir domainnya, atau kamu butuh
  data per-desa yang lebih presisi tanpa gantung ke API luar, admin
  BPBD bisa upload gambar peta topografi secara manual.
- Alih-alih admin mengetik "desa ini X mdpl" satu-satu, admin cukup
  UPLOAD GAMBAR peta topografi/kontur yang sudah berlabel nama
  kecamatan/desa & ada gradasi warna dataran rendah-sedang-tinggi.
- AI vision (lewat Groq) "membaca" gambar itu dan, untuk tiap wilayah
  yang diminta, mengeluarkan PERSENTASE proporsi warna
  Rendah / Sedang / Tinggi di area tsb -- BUKAN cuma 1 kategori rata
  untuk seluruh desa/kecamatan, karena topografi di dalam satu
  desa/kecamatan jarang seragam.

PENTING SOAL AKURASI (baca sebelum pakai di produksi):
Ini "Opsi A" -- AI membaca gambar apa adanya, tanpa tahu persis
gambar itu "nempel" di koordinat mana. Paling gampang buat admin
(tidak perlu kasih titik acuan koordinat), TAPI akurasinya bergantung
penuh pada:
  1. Apakah gambar punya label nama wilayah yang jelas & terbaca.
  2. Apakah legenda warnanya jelas / mengikuti konvensi umum
     (hijau/biru = rendah, kuning/oranye = sedang, merah/coklat/putih
     = tinggi).
  3. Kemampuan model vision yang dipakai untuk mengestimasi proporsi
     warna -- ini estimasi visual AI, BUKAN perhitungan piksel yang
     presisi secara matematis.
Karena itu, hasil AI di modul ini SELALU harus dianggap sebagai
DRAFT/PREVIEW yang wajib direview & bisa dikoreksi manual oleh admin
sebelum disimpan permanen (lihat route admin_topografi_analisis vs
admin_topografi_simpan di app.py -- dua langkah terpisah, bukan
langsung simpan).

Kalau ke depannya butuh akurasi lebih presisi & konsisten, pertimbangkan
"Opsi B": admin memberi titik acuan (lat/lon <-> pixel gambar), lalu
sistem melakukan pixel-sampling programatis (bukan AI) terhadap warna
gambar yang dioverlay ke file geojson kecamatan/desa yang sudah ada di
static/id3217_bandung_barat/.

PENYIMPANAN:
Hasil akhir (setelah direview/dikoreksi admin) disimpan ke Firebase
Realtime Database -- BUKAN file lokal -- karena hosting serverless
(Vercel) punya filesystem read-only. Pola ini sama dengan
'desa_overrides' yang sudah ada di app.py. Node yang dipakai:
  - topografi_manual_kec/{kecamatan_clean}
  - topografi_manual_desa/{kecamatan_clean}__{desa_clean}
"""

import os
import json
import re
import base64

STATIC_DIR_REL = os.path.join("static", "id3217_bandung_barat")
KECAMATAN_GEOJSON = "32.17_kecamatan.geojson"

# File-file yang bukan wilayah administratif desa/kecamatan (badan air dsb)
# -- dikecualikan dari daftar pilihan "per desa".
EXCLUDE_FILES = {KECAMATAN_GEOJSON, "id3217888_waduk.geojson"}


# ================================================================
# DAFTAR WILAYAH (buat dropdown di form admin)
# ================================================================

def _static_dir(app_root_path):
    return os.path.join(app_root_path, STATIC_DIR_REL)


def daftar_kecamatan(app_root_path):
    """
    Return list [{"nama": "Lembang", "clean": "lembang"}, ...] terurut
    abjad, dibaca dari file geojson batas kecamatan KBB.
    """
    from ml_engine import clean_name  # import lokal, hindari circular import

    path = os.path.join(_static_dir(app_root_path), KECAMATAN_GEOJSON)
    hasil = []

    if not os.path.exists(path):
        return hasil

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return hasil

    for feature in data.get("features", []):
        nama = str(feature.get("properties", {}).get("nm_kecamatan", "")).strip()
        if not nama:
            continue
        hasil.append({"nama": nama, "clean": clean_name(nama)})

    hasil.sort(key=lambda x: x["nama"])
    return hasil


def _peta_file_desa(app_root_path):
    """
    Bangun mapping {kecamatan_clean: path_file_geojson_desa}.

    Sengaja membaca properti 'district' dari ISI file geojson (bukan
    dari nama filenya), karena beberapa nama file ditulis beda dengan
    hasil clean_name() -- misalnya file
    'id3217140_cikalong_wetan.geojson' isinya district = "Cikalong Wetan",
    yang setelah clean_name jadi "cikalongwetan".
    """
    from ml_engine import clean_name

    folder = _static_dir(app_root_path)
    mapping = {}

    if not os.path.isdir(folder):
        return mapping

    for fname in os.listdir(folder):
        if fname in EXCLUDE_FILES or not fname.endswith(".geojson"):
            continue

        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            fitur = data.get("features", [])
            if not fitur:
                continue
            district = str(fitur[0].get("properties", {}).get("district", "")).strip()
            if not district:
                continue
            mapping[clean_name(district)] = fpath
        except Exception:
            continue

    return mapping


def daftar_desa(app_root_path, kecamatan_clean):
    """
    Return list [{"nama": "Cihampelas", "clean": "cihampelas"}, ...]
    untuk satu kecamatan (dipakai dropdown desa bertingkat).
    """
    from ml_engine import clean_name

    mapping = _peta_file_desa(app_root_path)
    fpath = mapping.get(kecamatan_clean)
    hasil = []

    if not fpath:
        return hasil

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return hasil

    for feature in data.get("features", []):
        nama = str(feature.get("properties", {}).get("village", "")).strip()
        if not nama:
            continue
        hasil.append({"nama": nama, "clean": clean_name(nama)})

    hasil.sort(key=lambda x: x["nama"])
    return hasil


# ================================================================
# KUNCI PENYIMPANAN (konsisten dengan pola desa_key di app.py)
# ================================================================

def kunci_desa(kecamatan_clean, desa_clean):
    return f"{kecamatan_clean}__{desa_clean}"


# ================================================================
# HELPER SKOR & KATEGORI DARI BREAKDOWN PERSENTASE
# ================================================================

def kategori_dominan_dari_breakdown(persen_rendah, persen_sedang, persen_tinggi):
    pasangan = [
        ("Dataran Rendah", persen_rendah),
        ("Dataran Sedang / Perbukitan", persen_sedang),
        ("Dataran Tinggi / Pegunungan", persen_tinggi),
    ]
    pasangan.sort(key=lambda x: x[1], reverse=True)
    return pasangan[0][0]


def skor_dari_breakdown(persen_rendah, persen_sedang, persen_tinggi):
    """
    Skor tertimbang dari breakdown persentase, memakai bobot yang SAMA
    dengan elevation_engine.skor_topografi() supaya kedua metode
    (API/fallback vs manual) tetap komparabel di skor risiko gabungan:
        Rendah = 0.0, Sedang = 0.5, Tinggi = 1.0
    """
    return round(
        (persen_rendah / 100.0) * 0.0
        + (persen_sedang / 100.0) * 0.5
        + (persen_tinggi / 100.0) * 1.0,
        3,
    )


def _clamp_persen(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, v))


def normalisasi_100(persen_rendah, persen_sedang, persen_tinggi):
    """Skalakan 3 angka supaya totalnya persis 100 (kalau totalnya > 0)."""
    pr, ps, pt = _clamp_persen(persen_rendah), _clamp_persen(persen_sedang), _clamp_persen(persen_tinggi)
    total = pr + ps + pt
    if total <= 0:
        return 0.0, 0.0, 0.0
    faktor = 100.0 / total
    return round(pr * faktor, 1), round(ps * faktor, 1), round(pt * faktor, 1)


# ================================================================
# PEMANGGILAN AI VISION (Groq) UNTUK BACA GAMBAR
# ================================================================

PROMPT_SISTEM = (
    "Kamu adalah asisten GIS untuk BPBD Kabupaten Bandung Barat (KBB), Indonesia. "
    "Tugasmu membaca SATU gambar peta topografi/ketinggian wilayah, lalu untuk "
    "SETIAP nama wilayah administratif yang diberikan di daftar, perkirakan "
    "PROPORSI (dalam persen, jumlah wajib 100) area wilayah itu yang termasuk "
    "kategori: "
    "'Rendah' (dataran rendah, kira-kira di bawah 500 mdpl, biasanya warna "
    "hijau/biru pada peta topografi umum), "
    "'Sedang' (dataran sedang/perbukitan, kira-kira 500-999 mdpl, biasanya "
    "warna kuning/oranye muda), "
    "'Tinggi' (dataran tinggi/pegunungan, kira-kira >=1000 mdpl, biasanya "
    "warna oranye tua/merah/coklat/putih). "
    "SANGAT PENTING: jangan asumsikan satu wilayah warnanya seragam -- kalau "
    "di gambar area itu terlihat campuran beberapa warna, berikan proporsi "
    "campurannya apa adanya, jangan dibulatkan jadi satu kategori saja. "
    "Kalau nama wilayah tidak terlihat jelas di gambar atau gambar tidak "
    "mencakup wilayah tsb, tetap sertakan nama wilayah itu dengan catatan "
    "\"tidak terlihat jelas di gambar, ini estimasi kasar\" dan isi persentase "
    "dengan estimasi terbaikmu (jangan kosong). "
    "Jawab HANYA dalam format JSON valid, tanpa teks lain, tanpa markdown "
    "code fence, dengan skema PERSIS seperti ini: "
    '{"hasil": [{"wilayah": "<nama sesuai daftar>", '
    '"persen_rendah": <angka 0-100>, "persen_sedang": <angka 0-100>, '
    '"persen_tinggi": <angka 0-100>, "catatan": "<catatan singkat>"}]}'
)


def _encode_image_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def _parse_json_longgar(teks):
    """Parse JSON dari balasan model, toleran terhadap markdown fence dsb."""
    teks = (teks or "").strip()
    teks = re.sub(r"^```(?:json)?", "", teks).strip()
    teks = re.sub(r"```$", "", teks).strip()

    try:
        return json.loads(teks)
    except Exception:
        pass

    awal = teks.find("{")
    akhir = teks.rfind("}")
    if awal != -1 and akhir != -1 and akhir > awal:
        try:
            return json.loads(teks[awal:akhir + 1])
        except Exception:
            return None

    return None


def analisis_gambar_dengan_ai(groq_client, model_name, image_bytes, mime_type, daftar_wilayah):
    """
    Kirim gambar + daftar nama wilayah ke model vision Groq, minta
    breakdown persentase Rendah/Sedang/Tinggi per wilayah.

    CATATAN: `model_name` harus model Groq yang mendukung input gambar
    (vision). Daftar model vision yang tersedia di Groq bisa berubah
    dari waktu ke waktu -- cek https://console.groq.com/docs/vision
    untuk model terbaru, lalu set lewat environment variable
    GROQ_VISION_MODEL kalau default di app.py sudah tidak berlaku.

    Return: (hasil_list, error)
        hasil_list: list of dict
            {wilayah, persen_rendah, persen_sedang, persen_tinggi,
             kategori_dominan, catatan}
        error: None kalau sukses, atau string pesan error kalau gagal
               total (hasil_list akan berupa list kosong kalau error).
    """
    if not groq_client:
        return [], "Groq API belum dikonfigurasi (GROQ_API_KEY kosong di server)."

    if not daftar_wilayah:
        return [], "Tidak ada wilayah yang dipilih untuk dianalisis."

    if not image_bytes:
        return [], "Gambar kosong / gagal dibaca."

    b64 = _encode_image_base64(image_bytes)
    data_url = f"data:{mime_type};base64,{b64}"
    daftar_teks = "\n".join(f"- {w}" for w in daftar_wilayah)

    try:
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": PROMPT_SISTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Daftar wilayah yang perlu diklasifikasi dari "
                                "gambar peta topografi berikut:\n" + daftar_teks
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=2000,
        )
    except Exception as e:
        return [], (
            f"Gagal memanggil model vision Groq ('{model_name}'): {e}. "
            "Kemungkinan model ini sudah tidak tersedia -- cek model vision "
            "terbaru di console.groq.com/docs/vision lalu set environment "
            "variable GROQ_VISION_MODEL."
        )

    mentah = ""
    try:
        mentah = completion.choices[0].message.content or ""
    except Exception:
        pass

    parsed = _parse_json_longgar(mentah)

    if not parsed or "hasil" not in parsed or not isinstance(parsed.get("hasil"), list):
        return [], (
            "AI tidak mengembalikan format JSON yang valid. Coba lagi, "
            "atau gunakan gambar dengan label wilayah & legenda warna yang "
            "lebih jelas."
        )

    hasil = []
    for item in parsed.get("hasil", []):
        if not isinstance(item, dict):
            continue
        nama = str(item.get("wilayah", "")).strip()
        if not nama:
            continue

        pr, ps, pt = normalisasi_100(
            item.get("persen_rendah", 0),
            item.get("persen_sedang", 0),
            item.get("persen_tinggi", 0),
        )

        hasil.append({
            "wilayah": nama,
            "persen_rendah": pr,
            "persen_sedang": ps,
            "persen_tinggi": pt,
            "kategori_dominan": kategori_dominan_dari_breakdown(pr, ps, pt),
            "catatan": str(item.get("catatan", "") or "").strip(),
        })

    if not hasil:
        return [], "AI tidak mengembalikan hasil untuk wilayah manapun dari gambar ini."

    return hasil, None
