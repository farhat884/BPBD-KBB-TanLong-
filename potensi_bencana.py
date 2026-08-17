"""
potensi_bencana.py
====================

Fitur "Potensi Bencana" -- skor gabungan berbasis rumus umum
manajemen risiko bencana (BNPB/InaRISK):

    Potensi Bencana = (Ancaman x Kerentanan) / Kapasitas

KOMPONEN RUMUS & SUMBER DATANYA DI APLIKASI INI:
- ANCAMAN (hazard): sekarang direpresentasikan sebagai DAFTAR JENIS
  BENCANA yang berpotensi terjadi di suatu desa (mis. Batujajar bisa
  dicentang "Banjir", "Tanah Longsor", dan "Gempa Bumi" sekaligus --
  artinya wilayah itu punya 3 jenis ancaman). Admin memilih jenis
  yang relevan lewat checklist di tab "Potensi Bencana" pada
  Dashboard Admin (bisa dicentang lebih dari satu, dan admin juga
  bisa menambah jenis bencana baru ke checklist kalau ada jenis yang
  belum tersedia). Kalau admin belum PERNAH mengisi checklist untuk
  suatu desa, skor Potensi Bencana desa itu tidak dihitung (dianggap
  "Data ancaman belum diisi") -- ditampilkan apa adanya, bukan
  ditebak/dikira-kira oleh sistem. Kalau admin sudah pernah menyimpan
  tapi tidak mencentang jenis apa pun, itu dianggap valid (memang
  desa tsb dinilai tidak punya ancaman bencana yang menonjol).
- KERENTANAN (vulnerability): pakai Persen_Rentan_Desa / _Kec yang
  sudah ada (kelompok rentan: balita/lansia, warga miskin,
  disabilitas -- lihat GLOSARIUM_EDUKASI['rentan'] di app.py).
- KAPASITAS (capacity): pakai skor Destana (Desa/Kecamatan Tangguh
  Bencana) yang sudah ada (get_destana_record() di app.py) -- makin
  tinggi skor Destana (persentase indikator kesiapsiagaan yang
  terpenuhi), makin besar kapasitas wilayah meredam potensi bencana.

CATATAN SKALA & AMBANG BATAS:
Karena belum ada standar baku resmi dari BPBD KBB untuk formula ini,
skala di bawah dipilih supaya masuk akal & mudah dibaca. Kalau ke
depannya BPBD punya matriks/ambang batas resmi sendiri (mis.
mengikuti pedoman BNPB), cukup sesuaikan konstanta & fungsi di file
ini -- pemanggilnya (app.py) tidak perlu diubah.
  - Ancaman dinormalisasi ke 0..1 = (jumlah jenis bencana yang
    dicentang untuk desa itu) / (jumlah TOTAL jenis bencana yang
    tersedia di checklist saat ini). Karena admin bisa menambah jenis
    baru ke checklist, "total" ini DINAMIS -- selalu dihitung ulang
    dari daftar jenis yang berlaku saat itu (lihat
    get_daftar_jenis_ancaman() di app.py), bukan angka tetap.
  - Kerentanan dinormalisasi ke 0..1 (persen rentan / 100).
  - Kapasitas dinormalisasi ke 0..1 (skor Destana / 100), dengan
    NILAI LANTAI (KAPASITAS_MINIMUM_PERSEN) supaya desa dengan skor
    Destana 0% tidak menyebabkan pembagian mendekati nol / skor
    meledak tak wajar -- tetap logis karena desa TANPA kesiapan sama
    sekali memang seharusnya berpotensi sangat tinggi.
  - Skor akhir = (Ancaman_norm x Kerentanan_norm / Kapasitas_norm)
    x 100, dibatasi (clamp) ke rentang 0-100.
  - Kategori: skor >= 67 "Tinggi", >= 34 "Sedang", selain itu
    "Rendah" (pola 3 tingkat ini konsisten dengan kategori_rentan()
    di ml_engine.py & kategori topografi yang sudah ada).
"""

import re

# ------------------------------------------------------------------
# DAFTAR JENIS ANCAMAN BAWAAN (bisa ditambah admin lewat Firebase --
# lihat get_daftar_jenis_ancaman() di app.py, yang menggabungkan
# daftar bawaan ini dengan jenis custom yang ditambahkan admin).
# ------------------------------------------------------------------
DEFAULT_JENIS_ANCAMAN = [
    {"id": "banjir", "label": "Banjir", "icon": "🌊", "custom": False},
    {"id": "tanah_longsor", "label": "Tanah Longsor", "icon": "⛰️", "custom": False},
    {"id": "gempa_bumi", "label": "Gempa Bumi", "icon": "🌍", "custom": False},
    {"id": "letusan_gunung_berapi", "label": "Letusan Gunung Berapi", "icon": "🌋", "custom": False},
    {"id": "angin_puting_beliung", "label": "Angin Puting Beliung", "icon": "🌪️", "custom": False},
    {"id": "kekeringan", "label": "Kekeringan", "icon": "☀️", "custom": False},
    {"id": "kebakaran_hutan_lahan", "label": "Kebakaran Hutan/Lahan", "icon": "🔥", "custom": False},
]

DEFAULT_JENIS_ANCAMAN_IDS = {j["id"] for j in DEFAULT_JENIS_ANCAMAN}

ICON_JENIS_ANCAMAN_CUSTOM = "⚠️"

BATAS_KATEGORI_TINGGI = 67.0
BATAS_KATEGORI_SEDANG = 34.0

# Lantai kapasitas (dalam persen) supaya rumus tidak dibagi nol /
# meledak tak wajar saat skor Destana suatu desa masih 0%.
KAPASITAS_MINIMUM_PERSEN = 5.0


def slugify_jenis_id(label):
    """Ubah label jenis ancaman (input bebas dari admin) jadi id
    yang aman dipakai sebagai key Firebase & value checkbox HTML."""
    label_bersih = str(label or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", label_bersih).strip("_")
    return slug or "jenis_tanpa_nama"


def label_skor_ancaman(jumlah_jenis, total_jenis):
    """Ubah jumlah jenis ancaman yang dicentang jadi teks ringkas,
    mis. '3 dari 7 jenis'. Return '-' kalau belum diisi (None)."""
    if jumlah_jenis is None:
        return "-"
    total = max(1, int(total_jenis or 0))
    return f"{int(jumlah_jenis)} dari {total} jenis"


def kategori_potensi_bencana(skor_100):
    """Klasifikasi skor Potensi Bencana (0-100) jadi Rendah/Sedang/Tinggi."""
    if skor_100 is None:
        return None
    if skor_100 >= BATAS_KATEGORI_TINGGI:
        return "Tinggi"
    elif skor_100 >= BATAS_KATEGORI_SEDANG:
        return "Sedang"
    else:
        return "Rendah"


def _to_float_aman(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def hitung_potensi_bencana(jenis_ancaman_terpilih, total_jenis_ancaman, persen_rentan, skor_kapasitas):
    """
    Hitung skor Potensi Bencana = (Ancaman x Kerentanan) / Kapasitas.

    Parameter
    ---------
    jenis_ancaman_terpilih : list[str]|None
        List id jenis bencana yang dicentang admin untuk desa ini
        (mis. ['banjir', 'tanah_longsor', 'gempa_bumi']). None kalau
        admin BELUM PERNAH mengisi checklist untuk wilayah ini --
        dalam kondisi ini fungsi TIDAK menghitung skor (menghindari
        asumsi/tebakan), cukup menandai 'lengkap': False. List KOSONG
        ([]) dianggap valid (sudah pernah diisi, memang tidak ada
        jenis ancaman yang dicentang).
    total_jenis_ancaman : int
        Jumlah TOTAL jenis bencana yang tersedia di checklist saat
        ini (termasuk jenis custom tambahan admin) -- dipakai sebagai
        pembagi normalisasi Ancaman.
    persen_rentan : int|float
        Persentase kelompok rentan (0-100), dari Persen_Rentan_Desa
        atau Persen_Rentan_Kec.
    skor_kapasitas : int|float
        Skor Destana (0-100), dari get_destana_record()['persen'].

    Return: dict
        {
          "skor": float|None,       # 0-100, 2 desimal (None kalau
                                     # 'lengkap' False)
          "kategori": str|None,     # 'Rendah'/'Sedang'/'Tinggi'
          "lengkap": bool,          # False kalau ancaman belum diisi
          "keterangan": str,        # pesan singkat kalau belum lengkap
          "jumlah_jenis": int|None, # jumlah jenis ancaman yang dicentang
          "total_jenis": int,       # total jenis yang tersedia saat ini
        }
    """
    total_jenis = max(1, int(_to_float_aman(total_jenis_ancaman, 0)))

    if jenis_ancaman_terpilih is None:
        return {
            "skor": None,
            "kategori": None,
            "lengkap": False,
            "keterangan": "Data ancaman belum diisi admin.",
            "jumlah_jenis": None,
            "total_jenis": total_jenis,
        }

    jumlah_jenis = len(jenis_ancaman_terpilih)

    ancaman_norm = max(0.0, min(1.0, jumlah_jenis / total_jenis))
    rentan_norm = max(0.0, min(100.0, _to_float_aman(persen_rentan))) / 100.0

    kapasitas_persen = max(0.0, min(100.0, _to_float_aman(skor_kapasitas)))
    kapasitas_norm = max(kapasitas_persen, KAPASITAS_MINIMUM_PERSEN) / 100.0

    skor_mentah = (ancaman_norm * rentan_norm / kapasitas_norm) * 100.0
    skor_final = round(max(0.0, min(100.0, skor_mentah)), 2)

    return {
        "skor": skor_final,
        "kategori": kategori_potensi_bencana(skor_final),
        "lengkap": True,
        "keterangan": "",
        "jumlah_jenis": jumlah_jenis,
        "total_jenis": total_jenis,
    }
