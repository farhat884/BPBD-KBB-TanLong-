"""
kajian_risiko.py
==================

Fitur "Kajian Risiko Bencana" -- tingkat risiko per JENIS BENCANA yang
dibaca LANGSUNG dari tabulasi resmi BPBD, bukan dihitung dari rumus.

Sumber data
-----------
File Excel `MATRIKS_KAJIAN_RISIKO_BENCANA_KAB__BANDUNGBARAT_V*.xlsx`
di folder `data/` (3 sheet: DESA, KECAMATAN, KABUPATEN). Tiap baris =
satu jenis bahaya di satu wilayah, lengkap dengan kelas bahaya,
kerentanan, kapasitas, dan KELAS RISIKO (Rendah/Sedang/Tinggi).

Kelas risiko yang ditampilkan di peta = kolom "RISIKO -> KELAS" di file
itu apa adanya. Modul ini TIDAK menghitung ulang / menebak nilai apa pun.

Mengganti data
--------------
Cukup taruh file matriks versi baru (nama diawali `MATRIKS_KAJIAN_RISIKO`)
di folder `data/`. Kalau ada beberapa file, yang dipakai adalah yang
namanya paling akhir secara alfabet (V3 > V2). Bisa juga dipaksa lewat
environment variable `MATRIKS_KAJIAN_PATH`. Posisi kolom dicari lewat
teks header (bukan nomor kolom), jadi tahan terhadap kolom yang
bergeser -- tapi kalau header penting hilang, file ditolak dengan pesan
jelas (fitur dinonaktifkan, aplikasi lain tetap jalan).
"""

import glob
import hashlib
import json
import os
import re

import pandas as pd

# ------------------------------------------------------------------
# DAFTAR JENIS BENCANA (urutan = urutan di dropdown)
# ------------------------------------------------------------------
# "Multi Bahaya" ditaruh paling atas karena jadi pilihan awal
# (gambaran gabungan seluruh bencana), sisanya alfabetis.
HAZARDS = [
    {"id": "multi", "label": "Multi Bahaya (gabungan)", "icon": "⚠️"},
    {"id": "banjir", "label": "Banjir", "icon": "🌊"},
    {"id": "banjir_bandang", "label": "Banjir Bandang", "icon": "🌊"},
    {"id": "cuaca_ekstrem", "label": "Cuaca Ekstrem", "icon": "🌪️"},
    {"id": "gempa_bumi", "label": "Gempa Bumi", "icon": "🌍"},
    {"id": "karhutla", "label": "Kebakaran Hutan dan Lahan", "icon": "🔥"},
    {"id": "kekeringan", "label": "Kekeringan", "icon": "☀️"},
    {"id": "letusan_gunung_api", "label": "Letusan Gunung Api", "icon": "🌋"},
    {"id": "tanah_longsor", "label": "Tanah Longsor", "icon": "⛰️"},
]

DEFAULT_HAZARD_ID = "multi"

# Nama bahaya di file Excel (huruf besar, sudah dirapikan) -> id.
# Sheet KABUPATEN memakai penulisan lain (GEMPABUMI, KARHUTLA), jadi
# dua-duanya didaftarkan di sini.
_HAZARD_ALIAS = {
    "MULTI": "multi",
    "BANJIR": "banjir",
    "BANJIR BANDANG": "banjir_bandang",
    "CUACA EKSTREM": "cuaca_ekstrem",
    "GEMPA BUMI": "gempa_bumi",
    "GEMPABUMI": "gempa_bumi",
    "KEBAKARAN HUTAN DAN LAHAN": "karhutla",
    "KARHUTLA": "karhutla",
    "KEKERINGAN": "kekeringan",
    "LETUSAN GUNUNG API": "letusan_gunung_api",
    "TANAH LONGSOR": "tanah_longsor",
}

HAZARD_BY_ID = {h["id"]: h for h in HAZARDS}

KELAS_VALID = ("Rendah", "Sedang", "Tinggi")

# Warna kelas risiko (mengikuti konvensi peta risiko: hijau/kuning/merah).
WARNA_KELAS = {
    "Rendah": "#1a9641",
    "Sedang": "#f9c74f",
    "Tinggi": "#d7191c",
}
WARNA_TIDAK_ADA = "#94a3b8"  # abu-abu: tidak tercatat di matriks

# Kadang nama desa di Excel salah ketik dibanding nama resmi di peta
# (GeoJSON). Pasangan (kunci kecamatan, kunci desa di Excel) -> kunci
# desa yang benar. Kalau nanti ada salah ketik baru, tambahkan di sini
# (atau lebih baik perbaiki langsung di Excel-nya).
ALIAS_DESA = {
    ("cililin", "nangerang"): "nanggerang",
    ("lembang", "cikoke"): "cikole",
    ("padalarang", "campakamekar"): "cempakamekar",
    ("sindangkerta", "bunianagara"): "buninagara",
}


# ------------------------------------------------------------------
# NORMALISASI NAMA
# ------------------------------------------------------------------
def key_js(nama):
    """
    Kunci nama yang SAMA PERSIS dengan cleanNameJS() di peta (JS):
    huruf kecil, buang kata 'kecamatan' & 'desa', buang semua selain
    a-z0-9. Dipakai untuk nama dari GeoJSON, supaya kunci yang dikirim
    ke JS cocok dengan yang dihitung JS dari properti fitur peta.
    """
    if nama is None:
        return ""
    s = str(nama).lower().replace("kecamatan", "").replace("desa", "")
    return re.sub(r"[^a-z0-9]", "", s)


def _key_excel(nama):
    """
    Kunci nama untuk teks di Excel: buang awalan 'Desa'/'Kelurahan'/
    'Kecamatan' (Excel menulis 'Desa Selacau', GeoJSON cuma 'Selacau'),
    lalu buang semua selain a-z0-9.
    """
    if nama is None or (isinstance(nama, float) and pd.isna(nama)):
        return ""
    s = str(nama).strip().lower()
    s = re.sub(r"^(desa|kelurahan|kel\.|kecamatan|kec\.)\s+", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _nama_rapi(nama):
    """Rapikan nama untuk tampilan: buang awalan 'Desa '/'Kelurahan '."""
    s = str(nama or "").strip()
    return re.sub(r"^(desa|kelurahan|kel\.)\s+", "", s, flags=re.IGNORECASE)


def _norm_hazard(teks):
    if teks is None or (isinstance(teks, float) and pd.isna(teks)):
        return None
    t = re.sub(r"\s+", " ", str(teks)).strip().upper()
    return _HAZARD_ALIAS.get(t)


def _num(v):
    """Angka aman: '-' / kosong / teks -> None."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.replace(",", "").strip()
        if v in ("", "-"):
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _kelas(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    t = str(v).strip().capitalize()
    return t if t in KELAS_VALID else None


# ------------------------------------------------------------------
# LOKASI KOLOM (dicari lewat teks header baris ke-4 di tiap sheet)
# ------------------------------------------------------------------
_BARIS_HEADER = 3  # indeks baris (0-based) yang memuat judul kolom utama

# judul header -> nama internal. Semuanya WAJIB ada.
_HEADER_WAJIB = {
    "JENIS BAHAYA": "hazard",
    "BAHAYA": "bahaya",  # kolom pertama dari 4 kolom luas bahaya
    "POTENSI PENDUDUK TERPAPAR (JIWA)": "penduduk",
    "POTENSI KERUGIAN (JUTA RUPIAH)": "kerugian_blok",
    "KELAS KERUGIAN": "kelas_kerugian",
    "KELAS KERENTANAN": "kelas_kerentanan",
    "KELAS KAPASITAS": "kelas_kapasitas",
    "RISIKO": "risiko",  # kolom pertama dari 4 kolom luas risiko
}


def _cari_kolom(df, level):
    """Petakan nama internal -> indeks kolom, dari header. Error jelas
    kalau ada header wajib yang tidak ditemukan."""
    header = df.iloc[_BARIS_HEADER].tolist()
    peta = {}
    for idx, teks in enumerate(header):
        if teks is None or (isinstance(teks, float) and pd.isna(teks)):
            continue
        t = re.sub(r"\s+", " ", str(teks)).strip().upper()
        if t in _HEADER_WAJIB and _HEADER_WAJIB[t] not in peta:
            peta[_HEADER_WAJIB[t]] = idx
        elif t == "KECAMATAN":
            peta.setdefault("kecamatan", idx)
        elif t in ("DESA/KELURAHAN", "DESA / KELURAHAN"):
            peta.setdefault("desa", idx)

    hilang = [h for h in _HEADER_WAJIB.values() if h not in peta]
    if level in ("kecamatan", "desa") and "kecamatan" not in peta:
        hilang.append("kecamatan")
    if level == "desa" and "desa" not in peta:
        hilang.append("desa")
    if hilang:
        raise ValueError(
            f"Sheet {level.upper()}: header kolom penting tidak ditemukan "
            f"({', '.join(sorted(hilang))}). Struktur file matriks berubah?"
        )

    b = peta["bahaya"]
    p = peta["penduduk"]
    k = peta["kerugian_blok"]
    r = peta["risiko"]
    return {
        "hazard": peta["hazard"],
        "kecamatan": peta.get("kecamatan"),
        "desa": peta.get("desa"),
        "luas_bahaya_total": b + 3,
        "kelas_bahaya": b + 4,
        "penduduk": p,
        "kerugian_total": k + 4,
        "kelas_kerentanan": peta["kelas_kerentanan"],
        "kelas_kapasitas": peta["kelas_kapasitas"],
        "luas_risiko_rendah": r,
        "luas_risiko_sedang": r + 1,
        "luas_risiko_tinggi": r + 2,
        "luas_risiko_total": r + 3,
        "kelas_risiko": r + 4,
    }


def _baca_sheet(df, level, warnings):
    """Ubah 1 sheet jadi list record. Baris yang jenis bahayanya tidak
    dikenal (mis. baris nomor kolom) dilewati; yang teks-nya aneh dicatat."""
    kol = _cari_kolom(df, level)
    hasil = []
    for i in range(_BARIS_HEADER + 1, len(df)):
        baris = df.iloc[i]
        mentah = baris.iloc[kol["hazard"]]
        hz = _norm_hazard(mentah)
        if hz is None:
            # baris nomor kolom (angka) / kosong = normal; teks lain = catat
            if isinstance(mentah, str) and mentah.strip():
                warnings.append(
                    f"Sheet {level.upper()} baris {i + 1}: jenis bahaya "
                    f"'{mentah}' tidak dikenali, baris dilewati."
                )
            continue

        kelas_risiko = _kelas(baris.iloc[kol["kelas_risiko"]])
        if kelas_risiko is None:
            warnings.append(
                f"Sheet {level.upper()} baris {i + 1}: kelas risiko kosong/"
                f"tidak valid, baris dilewati."
            )
            continue

        rec = {
            "hazard": hz,
            "r": kelas_risiko,
            "b": _kelas(baris.iloc[kol["kelas_bahaya"]]),
            "v": _kelas(baris.iloc[kol["kelas_kerentanan"]]),
            "c": _kelas(baris.iloc[kol["kelas_kapasitas"]]),
            "p": _num(baris.iloc[kol["penduduk"]]),
            "k": _num(baris.iloc[kol["kerugian_total"]]),
            "lr": _num(baris.iloc[kol["luas_risiko_total"]]),
        }
        if kol["kecamatan"] is not None:
            rec["kec"] = str(baris.iloc[kol["kecamatan"]]).strip()
        if kol["desa"] is not None:
            rec["desa"] = str(baris.iloc[kol["desa"]]).strip()
        hasil.append(rec)
    return hasil


# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------
def cari_file_matriks(folder_data):
    """Tentukan file matriks yang dipakai (env var > file terbaru di data/)."""
    env = os.environ.get("MATRIKS_KAJIAN_PATH", "").strip()
    if env:
        return env
    kandidat = sorted(
        glob.glob(os.path.join(folder_data, "MATRIKS_KAJIAN_RISIKO*.xlsx"))
    )
    return kandidat[-1] if kandidat else None


def load_matriks(path):
    """
    Baca file matriks -> dict:
        {
          "file": nama file,
          "desa": {(kunci_kec, kunci_desa): {hazard_id: record}},
          "kec":  {kunci_kec: {hazard_id: record}},
          "kab":  {hazard_id: record},
          "warnings": [str, ...],
        }
    Raise ValueError/FileNotFoundError kalau file/struktur tidak valid.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"File matriks kajian risiko tidak ditemukan: {path}")

    warnings = []
    sheets = pd.read_excel(
        path, sheet_name=["DESA", "KECAMATAN", "KABUPATEN"], header=None
    )

    desa, kec, kab = {}, {}, {}

    for rec in _baca_sheet(sheets["DESA"], "desa", warnings):
        kk = _key_excel(rec["kec"])
        dk = _key_excel(rec["desa"])
        alias = ALIAS_DESA.get((kk, dk))
        if alias:
            warnings.append(
                f"Nama desa di matriks '{rec['desa']}' (Kec. {rec['kec']}) "
                f"dianggap sama dengan '{alias}' di peta (salah ketik di Excel)."
            )
            dk = alias
        slot = desa.setdefault((kk, dk), {})
        if rec["hazard"] in slot:
            warnings.append(
                f"Duplikat baris: {rec['desa']} (Kec. {rec['kec']}) / "
                f"{rec['hazard']} -- baris terakhir yang dipakai."
            )
        slot[rec["hazard"]] = rec

    for rec in _baca_sheet(sheets["KECAMATAN"], "kecamatan", warnings):
        slot = kec.setdefault(_key_excel(rec["kec"]), {})
        slot[rec["hazard"]] = rec

    for rec in _baca_sheet(sheets["KABUPATEN"], "kabupaten", warnings):
        kab[rec["hazard"]] = rec

    # Peringatan (sekali) untuk alias yang berulang per-hazard: ringkas.
    warnings = list(dict.fromkeys(warnings))

    return {
        "file": os.path.basename(path),
        "desa": desa,
        "kec": kec,
        "kab": kab,
        "warnings": warnings,
    }


# ------------------------------------------------------------------
# CACHE JSON (supaya start server tidak perlu membaca Excel ~1,3 detik)
# ------------------------------------------------------------------
# Membaca xlsx 470 KB lewat pandas/openpyxl butuh sekitar 1,3 detik di
# SETIAP start server -- di hosting serverless itu terasa di tiap cold
# start. Hasil bacaannya karena itu disimpan sebagai JSON kecil
# (data/kajian_risiko_cache.json) yang ikut di-deploy dan dimuat dalam
# hitungan milidetik.
#
# Cache dianggap sah HANYA kalau sidik jari (SHA-1) file Excel-nya sama
# dengan yang tercatat di cache. Jadi kalau Excel diganti tapi cache lupa
# dibangun ulang, aplikasi otomatis membaca Excel (lebih lambat tapi
# tetap BENAR), dan mencoba menulis cache baru kalau foldernya bisa
# ditulis. Bangun ulang manual:  python kajian_risiko.py
_CACHE_VERSI = 1
_NAMA_CACHE = "kajian_risiko_cache.json"


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 16), b""):
            h.update(blok)
    return h.hexdigest()


def _simpan_cache(m, path_cache, sha1):
    data = {
        "versi": _CACHE_VERSI,
        "sha1": sha1,
        "file": m["file"],
        "warnings": m["warnings"],
        # kunci tuple tidak bisa jadi kunci JSON -> simpan sebagai daftar
        "desa": [[kk, dk, per] for (kk, dk), per in m["desa"].items()],
        "kec": m["kec"],
        "kab": m["kab"],
    }
    tmp = path_cache + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path_cache)  # atomik: tidak ada file setengah jadi


def _muat_cache(path_cache, sha1):
    """Return matriks dari cache, atau None kalau tidak ada/tidak cocok/rusak."""
    try:
        with open(path_cache, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("versi") != _CACHE_VERSI or data.get("sha1") != sha1:
            return None
        return {
            "file": data["file"],
            "desa": {(kk, dk): per for kk, dk, per in data["desa"]},
            "kec": data["kec"],
            "kab": data["kab"],
            "warnings": data.get("warnings", []),
            "dari_cache": True,
        }
    except Exception:  # noqa: BLE001 - cache rusak/hilang = baca Excel saja
        return None


def bangun_cache(root_path):
    """Baca Excel lalu (re)tulis cache. Dipanggil manual: python kajian_risiko.py"""
    path = cari_file_matriks(os.path.join(root_path, "data"))
    m = load_matriks(path)
    path_cache = os.path.join(root_path, "data", _NAMA_CACHE)
    _simpan_cache(m, path_cache, _sha1_file(path))
    return path_cache, m


def muat_matriks_default(root_path):
    """
    Muat matriks dari lokasi standar. TIDAK pernah raise: kalau gagal,
    return None dan cetak alasan (aplikasi tetap jalan, tab Kajian Risiko
    menampilkan info 'data belum tersedia').
    """
    try:
        path = cari_file_matriks(os.path.join(root_path, "data"))
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"File matriks kajian risiko tidak ditemukan: {path}"
            )

        sha1 = _sha1_file(path)
        path_cache = os.path.join(root_path, "data", _NAMA_CACHE)

        m = _muat_cache(path_cache, sha1)
        if m is None:
            m = load_matriks(path)  # lambat (~1,3 dtk) -- hanya kalau cache basi
            try:
                _simpan_cache(m, path_cache, sha1)
            except OSError:
                pass  # folder read-only (mis. serverless): abaikan
    except Exception as e:  # noqa: BLE001 - sengaja luas, jangan sampai app mati
        print(f"⚠️  Kajian Risiko Bencana dinonaktifkan: {type(e).__name__}: {e}")
        return None

    sumber = "cache" if m.get("dari_cache") else "Excel"
    print(
        f"✅ Kajian Risiko Bencana dimuat dari {m['file']} [{sumber}] "
        f"({len(m['desa'])} desa, {len(m['kec'])} kecamatan)."
    )
    for w in m["warnings"]:
        print(f"   ⚠️  {w}")
    return m


# ------------------------------------------------------------------
# DATA UNTUK PETA
# ------------------------------------------------------------------
def _ringkas(rec):
    """Record -> bentuk ringkas (kunci pendek) yang dikirim ke JS."""
    out = {"r": rec["r"]}
    for k in ("b", "v", "c"):
        if rec.get(k):
            out[k] = rec[k]
    if rec.get("p") is not None:
        out["p"] = int(round(rec["p"]))
    if rec.get("k") is not None:
        out["k"] = round(rec["k"], 2)
    if rec.get("lr") is not None:
        out["lr"] = round(rec["lr"], 2)
    return out


def _ringkas_semua(per_hazard):
    return {hid: _ringkas(rec) for hid, rec in per_hazard.items()}


def siapkan_data_peta(matriks, fitur_desa, fitur_kec, nama_desa_fn):
    """
    Bangun data JSON untuk JS peta.

    fitur_desa   : list fitur GeoJSON desa (sudah punya properti
                   'nm_kecamatan' dari load_local_geojson_files()).
    fitur_kec    : list fitur GeoJSON kecamatan.
    nama_desa_fn : fungsi(properties) -> nama desa (get_nama_desa di app.py).

    Kunci desa = "<kunci kecamatan>|<kunci desa>" (GABUNGAN, karena ada
    nama desa yang sama di kecamatan berbeda, mis. Mekarjaya, Cipada).

    Return dict:
        hazards : list {id,label,icon} untuk dropdown
        kab     : {hazard_id: ringkas}
        kec     : {kunci_kec: {"n": nama, "h": {hazard_id: ringkas}}}
        desa    : {"kec|desa": {"n": nama, "k": nama_kec, "h": {...}}}
        tersedia: bool
        laporan : {"desa_tanpa_data": [...], "matriks_tak_terpakai": [...]}
    """
    if matriks is None:
        return {
            "hazards": HAZARDS, "kab": {}, "kec": {}, "desa": {},
            "tersedia": False,
            "laporan": {"desa_tanpa_data": [], "matriks_tak_terpakai": []},
        }

    # nama kecamatan tampilan, dari GeoJSON kecamatan
    kec_out = {}
    for f in fitur_kec:
        nama = (f.get("properties") or {}).get("nm_kecamatan", "")
        kk = key_js(nama)
        if not kk:
            continue
        kec_out[kk] = {"n": str(nama).strip(), "h": _ringkas_semua(matriks["kec"].get(kk, {}))}

    desa_out = {}
    terpakai = set()
    tanpa_data = []
    for f in fitur_desa:
        props = f.get("properties") or {}
        nama = nama_desa_fn(props)
        kk = key_js(props.get("nm_kecamatan", ""))
        dk = key_js(nama)
        if not dk or dk == "waduk":
            continue
        per_hazard = matriks["desa"].get((kk, dk))
        if per_hazard:
            terpakai.add((kk, dk))
        else:
            tanpa_data.append(f"{nama} (Kec. {kec_out.get(kk, {}).get('n', kk)})")
        desa_out[f"{kk}|{dk}"] = {
            "n": nama,
            "k": kec_out.get(kk, {}).get("n", ""),
            "h": _ringkas_semua(per_hazard or {}),
        }

    tak_terpakai = sorted(
        f"{next(iter(v.values()))['desa']} (Kec. {next(iter(v.values()))['kec']})"
        for key, v in matriks["desa"].items()
        if key not in terpakai
    )

    return {
        "hazards": HAZARDS,
        "kab": _ringkas_semua(matriks["kab"]),
        "kec": kec_out,
        "desa": desa_out,
        "tersedia": True,
        "laporan": {
            "desa_tanpa_data": tanpa_data,
            "matriks_tak_terpakai": tak_terpakai,
        },
    }


def popup_ringkasan_html(per_hazard_ringkas):
    """
    Blok HTML ringkasan risiko SEMUA jenis bencana untuk satu desa
    (dipakai di popup klik desa). per_hazard_ringkas = {hazard_id: ringkas}.
    """
    warna_chip = {
        "Rendah": ("#dcfce7", "#166534"),
        "Sedang": ("#fef3c7", "#92400e"),
        "Tinggi": ("#fee2e2", "#991b1b"),
    }
    chips = []
    for hz in HAZARDS:
        rec = per_hazard_ringkas.get(hz["id"])
        if not rec:
            continue
        bg, fg = warna_chip[rec["r"]]
        chips.append(
            f'<span style="display:inline-block;margin:2px 3px 0 0;padding:1px 7px;'
            f'border-radius:9px;background:{bg};color:{fg};font-size:10px;">'
            f'{hz["icon"]} {hz["label"].split(" (")[0]}: <b>{rec["r"]}</b></span>'
        )
    if not chips:
        return (
            '<span style="font-size:10px;color:#7f8c8d;">'
            "<i>Tidak tercatat di matriks kajian risiko</i></span>"
        )
    return "".join(chips)


def opsi_dropdown_html():
    """<option> untuk dropdown pilih bencana (Multi Bahaya terpilih awal)."""
    baris = []
    for hz in HAZARDS:
        sel = " selected" if hz["id"] == DEFAULT_HAZARD_ID else ""
        baris.append(f'<option value="{hz["id"]}"{sel}>{hz["icon"]} {hz["label"]}</option>')
    return "".join(baris)


def data_js(data_peta):
    """Kumpulan placeholder -> JSON string untuk disisipkan ke script peta."""
    def dump(x):
        # separators rapat; ensure_ascii=True supaya aman di dalam srcdoc/HTML
        return json.dumps(x, separators=(",", ":"), ensure_ascii=True)

    return {
        "__KAJIAN_HAZARDS_JSON__": dump(data_peta["hazards"]),
        "__KAJIAN_KAB_JSON__": dump(data_peta["kab"]),
        "__KAJIAN_KEC_JSON__": dump(data_peta["kec"]),
        "__KAJIAN_DESA_JSON__": dump(data_peta["desa"]),
        "__KAJIAN_TERSEDIA__": "true" if data_peta["tersedia"] else "false",
        "__KAJIAN_WARNA_JSON__": dump({**WARNA_KELAS, "_none": WARNA_TIDAK_ADA}),
        "__KAJIAN_DEFAULT_HAZARD__": DEFAULT_HAZARD_ID,
    }


if __name__ == "__main__":
    # python kajian_risiko.py  -> bangun ulang data/kajian_risiko_cache.json
    _root = os.path.dirname(os.path.abspath(__file__))
    _p, _m = bangun_cache(_root)
    print(f"Cache ditulis: {_p} ({len(_m['desa'])} desa, {len(_m['kec'])} kecamatan)")
