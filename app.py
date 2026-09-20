import json
import os
import html
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from functools import wraps
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db
from groq import Groq
# rag_engine (chatbot / basis pengetahuan) memuat scikit-learn + scipy yang
# butuh ~0,9 detik hanya untuk di-import. Halaman login, dashboard, dan peta
# tidak memerlukannya, jadi dimuat MALAS: baru saat fungsi chatbot/pengetahuan
# pertama kali dipanggil. Nama & cara pakainya di seluruh app.py tidak berubah.
def _rag_malas(nama):
    def _panggil(*args, **kwargs):
        import rag_engine
        return getattr(rag_engine, nama)(*args, **kwargs)
    _panggil.__name__ = nama
    return _panggil


extract_text_bytes = _rag_malas("extract_text_bytes")
save_document = _rag_malas("save_document")
save_manual_knowledge = _rag_malas("save_manual_knowledge")
list_documents = _rag_malas("list_documents")
retrieve = _rag_malas("retrieve")
build_context = _rag_malas("build_context")
delete_document = _rag_malas("delete_document")

import folium

from ml_engine import get_ml_clustered_data, clean_name, kategori_rentan, update_desa_excel, bagikan_kuota_sisa_terbesar
import kajian_risiko
import json
import os
import re
import secrets
import time
import threading
from functools import lru_cache

# Muat variabel dari file .env (kalau ada) ke environment lokal.
# Wajib dipanggil SEBELUM os.getenv/os.environ.get dipakai di bawah,
# supaya GROQ_API_KEY, FIREBASE_CONFIG_JSON, dll bisa terbaca.
load_dotenv()

#bpbd_kbb_super_rahasia_@2024_jangan_bocor

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-key-hanya-untuk-lokal')


# ========================================================
# KOMPRESI GZIP
# ========================================================
# Halaman peta (/kondisi) ~7 MB mentah tapi ~1,2 MB kalau di-gzip (teks
# HTML/JSON sangat mudah dikompres), dan dashboard admin ~0,9 MB. Tanpa
# kompresi, semuanya dikirim mentah -- lambat di jaringan seluler. Hook ini
# mengompres respons teks bila browser mengaku mendukung gzip. Kalau
# hosting sudah mengompres sendiri (header Content-Encoding sudah ada),
# respons dilewatkan tanpa diubah.
import gzip as _gzip

_MIME_KOMPRESIBEL = (
    "text/html", "text/css", "text/plain", "text/javascript",
    "application/javascript", "application/json", "application/geo+json",
    "image/svg+xml",
)
_UKURAN_MIN_GZIP = 1024  # di bawah ini tak sebanding dengan biayanya


@app.after_request
def kompres_gzip(response):
    try:
        if (
            response.status_code != 200
            or response.direct_passthrough
            or response.is_streamed
            or "Content-Encoding" in response.headers
            or "gzip" not in request.headers.get("Accept-Encoding", "").lower()
            or response.mimetype not in _MIME_KOMPRESIBEL
        ):
            return response

        data = response.get_data()
        if len(data) < _UKURAN_MIN_GZIP:
            return response

        response.set_data(_gzip.compress(data, compresslevel=5, mtime=0))
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(response.get_data()))
        response.headers.add("Vary", "Accept-Encoding")
    except Exception as e:  # noqa: BLE001 - kompresi tak boleh menjatuhkan request
        print(f"⚠️  Kompresi gzip dilewati: {e}")
    return response

_groq_api_key = os.getenv("GROQ_API_KEY")
if not _groq_api_key:
    print(
        "⚠️  GROQ_API_KEY tidak ditemukan di environment/.env. "
        "Fitur chatbot AI fallback (Groq) akan dinonaktifkan, "
        "tapi dashboard & fitur lain tetap bisa jalan."
    )
groq_client = Groq(api_key=_groq_api_key) if _groq_api_key else None

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
GROQ_SAFE_FALLBACKS = [
    GROQ_MODEL,
    GROQ_FALLBACK_MODEL,
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

def get_available_groq_models():
    """Ambil model yang benar-benar tersedia untuk API key/project ini."""
    if groq_client is None:
        return []
    try:
        result = groq_client.models.list()
        return [getattr(m, "id", "") for m in getattr(result, "data", []) if getattr(m, "id", "")]
    except Exception as exc:
        print(f"[AI/RAG] Gagal membaca daftar model Groq: {type(exc).__name__}: {exc}")
        return []


# ========================================================
# SUPABASE - DOKUMEN KNOWLEDGE BASE
# ========================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "ai-guides").strip() or "ai-guides"
SUPABASE_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

def get_supabase_admin():
    """Client backend menggunakan Secret Key. Jangan pernah dikirim ke browser."""
    if not SUPABASE_URL or not os.getenv("SUPABASE_SECRET_KEY"):
        raise RuntimeError("SUPABASE_URL/SUPABASE_SECRET_KEY belum tersedia.")
    from supabase import create_client
    return create_client(SUPABASE_URL, os.getenv("SUPABASE_SECRET_KEY"))

# ========================================================
# KONEKSI KE FIREBASE
# ========================================================
if not firebase_admin._apps:
    # Cek apakah ada environment variable kredensial di Vercel
    firebase_json_env = os.environ.get('FIREBASE_CONFIG_JSON')
    
    if firebase_json_env:
        # Menggunakan environment variable dari Vercel
        cred_dict = json.loads(firebase_json_env)
        cred = credentials.Certificate(cred_dict)
    elif os.path.exists('firebase_key.json'):
        # Fallback untuk running lokal
        cred = credentials.Certificate('firebase_key.json')
    else:
        raise FileNotFoundError("Kredensial Firebase tidak ditemukan.")

    firebase_admin.initialize_app(cred, {
        'databaseURL': os.environ.get('FIREBASE_DB_URL', 'https://bpbd-kbb-default-rtdb.asia-southeast1.firebasedatabase.app/')
    })

print("Berhasil terhubung ke Firebase!")

# ========================================================
# PREFETCH FIREBASE PARALEL SAAT START
# ========================================================
# Saat start, aplikasi membaca 3 node Firebase yang saling bebas. Dulu
# dibaca satu per satu (kali tunggu jaringan berurutan) SESUDAH semua
# pekerjaan lokal selesai. Sekarang semuanya diminta serentak di thread
# latar SEKARANG JUGA, jadi menunggu jaringannya tumpang-tindih dengan
# pekerjaan lokal (baca Excel, klasterisasi, dst) dan satu sama lain.
#
# _fb_get(path) memberi hasil prefetch itu kalau masih segar (<= 30 detik,
# artinya memang hasil start ini); selain itu -- termasuk pemanggilan
# berikutnya setelah server berjalan lama -- membaca Firebase langsung
# seperti biasa. Kalau prefetch gagal, error yang SAMA dilempar ulang ke
# pemanggil, sehingga try/except di tiap fungsi sync berperilaku persis
# seperti sebelumnya.
_FB_PREFETCH_PATHS = (
    'desa_overrides',
    'kajian_risiko',
    'laporan_edukasi',
)
_FB_PREFETCH_MAX_UMUR = 30  # detik
_FB_PREFETCH = {}


def _mulai_prefetch_firebase():
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(
        max_workers=len(_FB_PREFETCH_PATHS),
        thread_name_prefix="fb-prefetch",
    )
    waktu = time.time()
    for path in _FB_PREFETCH_PATHS:
        _FB_PREFETCH[path] = (
            waktu,
            executor.submit(lambda p=path: db.reference(p).get()),
        )
    executor.shutdown(wait=False)  # thread selesai sendiri setelah tugasnya


def _fb_get(path):
    """db.reference(path).get() -- pakai hasil prefetch start kalau masih segar."""
    item = _FB_PREFETCH.pop(path, None)
    if item is not None:
        waktu, future = item
        if time.time() - waktu <= _FB_PREFETCH_MAX_UMUR:
            return future.result()  # lempar ulang error aslinya bila gagal
    return db.reference(path).get()


_mulai_prefetch_firebase()

# ========================================================
# AUTH HELPERS (Login & Role)
# ========================================================

def login_required(f):
    """Halaman hanya bisa diakses jika sudah login (role apapun)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'email' not in session:
            flash("Silakan login terlebih dahulu.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    """Halaman hanya bisa diakses oleh role tertentu, misal role_required('admin')."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'email' not in session:
                flash("Silakan login terlebih dahulu.")
                return redirect(url_for('login'))
            if session.get('role') not in roles:
                return render_template("403.html"), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

 

# ========================================================
# CSRF PROTECTION UNTUK AKSI ADMIN
# ========================================================

def get_csrf_token():
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


def validate_csrf():
    token_session = session.get('_csrf_token')
    token_form = request.form.get('_csrf_token', '')
    if not token_form and request.is_json:
        payload = request.get_json(silent=True) or {}
        token_form = payload.get('_csrf_token', '')
    if not token_session or not token_form or not secrets.compare_digest(token_session, token_form):
        return False
    return True


@app.context_processor
def inject_security_helpers():
    return {'csrf_token': get_csrf_token()}

# ========================================================
# PREPARE DATA
# ========================================================
 
df_data = get_ml_clustered_data(app.root_path)
 
desa_dict = {}
kec_dict = {}
 
 
def safe_number(value, default=0):
    """
    Mengubah nilai menjadi angka dengan aman.
    Bisa menangani:
    - int
    - float
    - string angka
    - string dengan koma
    - None
    """
    if value is None:
        return default
 
    try:
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
 
        return float(value)
 
    except (ValueError, TypeError):
        return default
 
 
if not df_data.empty:
 
    print("\n========================================")
    print("MEMPROSES DATA DESA & KECAMATAN")
    print("========================================")
 
    # ----------------------------------------------------
    # CEK KOLOM DATA
    # ----------------------------------------------------
 
    print("Kolom data yang tersedia:")
    print(list(df_data.columns))
 
    # ----------------------------------------------------
    # LOOP DATA
    # ----------------------------------------------------
 
    for row in df_data.to_dict("records"):
 
        # -----------------------------------------------
        # NAMA DESA
        # -----------------------------------------------
 
        raw_desa = row.get("Desa", "")
        clean_d = clean_name(raw_desa)
 
        # -----------------------------------------------
        # NAMA KECAMATAN
        # -----------------------------------------------
 
        raw_kec = row.get("Kecamatan", "")
        clean_k = clean_name(raw_kec)
 
        if not clean_d:
            continue
 
        # -----------------------------------------------
        # DATA DASAR DESA
        # -----------------------------------------------
        #
        # CATATAN: "Warga_Wajib_Edukasi_Desa" berasal dari
        # kolom jumlah_penduduk di data sumber. Kolom ini
        # BUKAN total penduduk desa, melainkan jumlah warga
        # yang wajib diedukasi di desa tsb.
        # -----------------------------------------------
 
        warga_wajib_edukasi_desa = safe_number(
            row.get("Warga_Wajib_Edukasi_Desa", 0)
        )
 
        # -----------------------------------------------
        # DETAIL KELOMPOK RENTAN DESA
        # -----------------------------------------------
 
        rentan_balita_lansia_desa = safe_number(
            row.get("Rentan_BalitaLansia_Desa", 0)
        )
        rentan_miskin_desa = safe_number(
            row.get("Rentan_Miskin_Desa", 0)
        )
        rentan_disabilitas_desa = safe_number(
            row.get("Rentan_Disabilitas_Desa", 0)
        )
        total_rentan_desa = safe_number(
            row.get("Total_Rentan_Desa", 0)
        )
        persen_rentan_desa = safe_number(
            row.get("Persen_Rentan_Desa", 0)
        )
        kategori_rentan_desa = row.get(
            "Kategori_Rentan_Desa", "-"
        )
 
        # -----------------------------------------------
        # TOTAL TERPAPAR (WAJIB EDUKASI) KECAMATAN
        # -----------------------------------------------
 
        Terpapar_Kecamatan = safe_number(
            row.get("Terpapar_Kecamatan", 0)
        )
 
        # -----------------------------------------------
        # PERSENTASE TEREDUKASI DESA (penentu warna desa)
        # -----------------------------------------------
        #
        # Rumus:
        #
        # warga wajib edukasi desa
        # ------------------------- x 100
        # warga terpapar kecamatan
        #
        # -----------------------------------------------
 
        persen_teredukasi_desa = safe_number(
            row.get("Persen_Teredukasi_Desa", 0)
        )
 
        # -----------------------------------------------
        # SIMPAN DATA DESA
        # -----------------------------------------------
 
        desa_data = dict(row)
 
        desa_data["Warga_Wajib_Edukasi_Desa"] = warga_wajib_edukasi_desa
        desa_data["Rentan_BalitaLansia_Desa"] = rentan_balita_lansia_desa
        desa_data["Rentan_Miskin_Desa"] = rentan_miskin_desa
        desa_data["Rentan_Disabilitas_Desa"] = rentan_disabilitas_desa
        desa_data["Total_Rentan_Desa"] = total_rentan_desa
        desa_data["Persen_Rentan_Desa"] = persen_rentan_desa
        desa_data["Kategori_Rentan_Desa"] = kategori_rentan_desa
        desa_data["Terpapar_Kecamatan"] = Terpapar_Kecamatan
        desa_data["Persen_Teredukasi_Desa"] = round(
            persen_teredukasi_desa,
            2
        )
 

        # -----------------------------------------------
        # REALISASI EDUKASI AKTUAL (dari laporan petugas)
        # -----------------------------------------------
        # Field ini BEDA dengan "Persen_Teredukasi_Desa" di atas
        # (yang sebenarnya adalah persentase PRIORITAS warga,
        # bukan realisasi edukasi sungguhan). Nilai di sini diisi
        # belakangan oleh sync_realisasi_edukasi() berdasarkan
        # akumulasi laporan edukasi petugas yang tersimpan di
        # Firebase, lalu di-update live setiap ada laporan baru.
        # -----------------------------------------------
        desa_data["Jumlah_Teredukasi_Aktual_Desa"] = 0
        desa_data["Persen_Realisasi_Edukasi_Desa"] = 0.0

        desa_dict[clean_d] = desa_data
 
        # -----------------------------------------------
        # DATA KECAMATAN
        # -----------------------------------------------
 
        if clean_k not in kec_dict:
 
            kec_dict[clean_k] = {
 
                "Kecamatan": raw_kec,
 
                "Total_Warga_Wajib_Edukasi_Kecamatan":
                    safe_number(
                        row.get(
                            "Total_Warga_Wajib_Edukasi_Kecamatan",
                            0
                        )
                    ),
 
                "Terpapar_Kecamatan":
                    Terpapar_Kecamatan,
 
                "Rentan_BalitaLansia_Kec":
                    safe_number(
                        row.get("Rentan_BalitaLansia_Kec", 0)
                    ),
 
                "Rentan_Disabilitas_Kec":
                    safe_number(
                        row.get("Rentan_Disabilitas_Kec", 0)
                    ),
 
                "Rentan_IbuHamil_Kec":
                    safe_number(
                        row.get("Rentan_IbuHamil_Kec", 0)
                    ),
 
                "Total_Rentan_Kec":
                    safe_number(
                        row.get("Total_Rentan_Kec", 0)
                    ),
 
                "Persen_Rentan_Kec":
                    safe_number(
                        row.get("Persen_Rentan_Kec", 0)
                    ),
 
                "Kategori_Rentan_Kec":
                    row.get("Kategori_Rentan_Kec", "-"),
 
                "Kelas_Risiko_Kec":
                    row.get("Kelas_Risiko_Kec", "-"),

                # -----------------------------------------------
                # FAKTOR TOPOGRAFI (dataran tinggi/rendah) +
                # SKOR RISIKO GABUNGAN (Kelas_Risiko_Kec asli
                # yang sudah ditambah bobot ketinggian wilayah)
                # -----------------------------------------------
                "Elevasi_M_Kec":
                    row.get("Elevasi_M_Kec"),

                "Kategori_Topografi_Kec":
                    row.get("Kategori_Topografi_Kec", "-"),

                "Skor_Risiko_Gabungan_Kec":
                    safe_number(
                        row.get("Skor_Risiko_Gabungan_Kec", 0)
                    ),

                "Kategori_Risiko_Gabungan_Kec":
                    row.get("Kategori_Risiko_Gabungan_Kec", "-"),
 
                "Persen_Teredukasi_Kecamatan":
                    safe_number(
                        row.get(
                            "Persen_Teredukasi_Kecamatan",
                            0
                        )
                    ),

                # Realisasi edukasi aktual (diisi belakangan oleh
                # sync_realisasi_edukasi(), sama seperti di desa_data)
                "Jumlah_Teredukasi_Aktual_Kec": 0,
                "Persen_Realisasi_Edukasi_Kec": 0.0
            }
 
 
    # ====================================================
    # DEBUG DATA
    # ====================================================
 
    print("\n========================================")
    print("HASIL DATA TERPAPAR (WAJIB EDUKASI) KECAMATAN")
    print("========================================")
 
    for key, data in kec_dict.items():
 
        print(
            f"Kecamatan: "
            f"{data['Kecamatan']}"
        )
 
        print(
            f"Total Terpapar (Wajib Edukasi): "
            f"{data['Terpapar_Kecamatan']}"
        )
 
        print(
            f"Kelompok Rentan Kecamatan: "
            f"{data['Total_Rentan_Kec']} "
            f"({data['Persen_Rentan_Kec']}% - "
            f"{data['Kategori_Rentan_Kec']})"
        )
 
        print(
            f"Persentase Prioritas Warga yang Teredukasi (Kecamatan): "
            f"{data['Persen_Teredukasi_Kecamatan']}%"
        )
 
        print("----------------------------------------")
 
 
    print("\n========================================")
    print("HASIL PERHITUNGAN DESA")
    print("========================================")
 
    for key, data in desa_dict.items():
 
        print(
            f"Desa: {data.get('Desa', '-')}"
        )
 
        print(
            f"Kecamatan: "
            f"{data.get('Kecamatan', '-')}"
        )
 
        print(
            f"Warga Wajib Edukasi: "
            f"{data.get('Warga_Wajib_Edukasi_Desa', 0)}"
        )
 
        print(
            f"Kelompok Rentan Desa: "
            f"{data.get('Total_Rentan_Desa', 0)} "
            f"({data.get('Persen_Rentan_Desa', 0)}% - "
            f"{data.get('Kategori_Rentan_Desa', '-')})"
        )
 
        print(
            f"Terpapar Kecamatan: "
            f"{data.get('Terpapar_Kecamatan', 0)}"
        )
 
        print(
            f"Persentase Prioritas Warga yang Teredukasi (Desa): "
            f"{data.get('Persen_Teredukasi_Desa', 0)}%"
        )
 
        print("----------------------------------------")


# ========================================================
# REALISASI EDUKASI AKTUAL (dari laporan petugas di Firebase)
# ========================================================
#
# CATATAN:
# "Persen_Teredukasi_Desa" / "Persen_Teredukasi_Kecamatan" di atas
# BUKAN persentase realisasi edukasi sungguhan -- itu adalah
# "Persentase Prioritas Warga yang Teredukasi", yaitu pangsa/bobot
# target warga wajib edukasi desa ini dibanding warga terpapar di
# kecamatannya (angka statis dari data Excel, dipakai untuk
# menentukan prioritas wilayah mana yang perlu diedukasi duluan).
#
# Fungsi-fungsi di bawah ini menghitung metrik yang BERBEDA, yaitu
# REALISASI: berapa persen dari target warga wajib edukasi di
# desa/kecamatan tsb yang SUDAH benar-benar diedukasi, dihitung dari
# akumulasi laporan kegiatan edukasi yang diinput petugas (tersimpan
# di Firebase node 'laporan_edukasi').
# ========================================================


# ========================================================
# EDIT DATA DESA (via panel admin)
# ========================================================
#
# PENTING -- kenapa disimpan ke Firebase, BUKAN ke file Excel:
# Di hosting serverless seperti Vercel, folder deploy (/var/task)
# bersifat READ-ONLY. Server TIDAK BISA menulis balik ke file Excel
# sumber (data/Data_Desa_Longsor.xlsx) -- itu akan selalu gagal
# dengan "OSError: Read-only file system". Karena itu, perubahan
# dari admin disimpan sebagai "override" di Firebase node
# 'desa_overrides/{desa_key}', lalu diterapkan ULANG ke desa_dict
# setiap kali server start (lihat sync_desa_overrides() di bawah).
# Pola ini sama seperti laporan_edukasi yang sudah ada.
# ========================================================

def terapkan_data_desa(desa_key, warga, bl, miskin, disabilitas):
    """
    Terapkan nilai (warga wajib edukasi + 3 kategori rentan) ke
    desa_dict untuk satu desa, lalu hitung ulang field turunannya
    (Total_Rentan_Desa, Persen_Rentan_Desa, Kategori_Rentan_Desa,
    Persen_Teredukasi_Desa) beserta agregat kecamatan induknya
    (Total_Warga_Wajib_Edukasi_Kecamatan).

    Dipakai dari 2 tempat:
        1. Route admin_edit_desa() -- saat admin submit form edit.
        2. sync_desa_overrides() -- saat server start, menerapkan
           ulang semua override tersimpan dari Firebase.

    Return True kalau desa_key ditemukan & berhasil diterapkan.
    """

    if desa_key not in desa_dict:
        return False

    data = desa_dict[desa_key]

    total_rentan = bl + miskin + disabilitas
    persen_rentan = round((total_rentan / warga * 100), 2) if warga > 0 else 0.0

    data['Warga_Wajib_Edukasi_Desa'] = warga
    data['Rentan_BalitaLansia_Desa'] = bl
    data['Rentan_Miskin_Desa'] = miskin
    data['Rentan_Disabilitas_Desa'] = disabilitas
    data['Total_Rentan_Desa'] = total_rentan
    data['Persen_Rentan_Desa'] = persen_rentan
    data['Kategori_Rentan_Desa'] = kategori_rentan(persen_rentan)

    terpapar_kec = data.get('Terpapar_Kecamatan', 0)
    data['Persen_Teredukasi_Desa'] = round(
        (warga / terpapar_kec * 100), 2
    ) if terpapar_kec > 0 else 0.0

    # Kecamatan induk: target warga wajib edukasi kecamatan adalah
    # jumlah dari seluruh desa di kecamatan tsb, jadi perlu dihitung
    # ulang tiap kali salah satu desanya berubah.
    key_kec = clean_name(data.get('Kecamatan', ''))
    if key_kec in kec_dict:
        total_target_kec = sum(
            dd.get('Warga_Wajib_Edukasi_Desa', 0)
            for dd in desa_dict.values()
            if clean_name(dd.get('Kecamatan', '')) == key_kec
        )
        kec_dict[key_kec]['Total_Warga_Wajib_Edukasi_Kecamatan'] = total_target_kec

    return True


def sync_desa_overrides():
    """
    Baca semua perubahan data desa yang pernah disimpan admin
    (Firebase node 'desa_overrides') lalu terapkan ke desa_dict.
    Dipanggil sekali saat server start, SEBELUM sync_realisasi_edukasi()
    -- supaya saat persentase realisasi dihitung, ia sudah memakai
    angka 'warga wajib edukasi' hasil edit admin, bukan angka asli
    dari Excel.
    """

    try:
        override_data = _fb_get('desa_overrides') or {}
    except Exception as e:
        print(f"⚠️  Gagal sinkronisasi override data desa dari Firebase: {e}")
        override_data = {}

    diterapkan = 0

    for desa_key, ov in override_data.items():
        if not isinstance(ov, dict):
            continue

        berhasil = terapkan_data_desa(
            desa_key,
            safe_number(ov.get('warga', 0)),
            safe_number(ov.get('bl', 0)),
            safe_number(ov.get('miskin', 0)),
            safe_number(ov.get('disabilitas', 0)),
        )

        if berhasil:
            diterapkan += 1

    print(
        f"✅ Override data desa tersinkron dari Firebase "
        f"({diterapkan} dari {len(override_data)} entri diterapkan)."
    )


# Terapkan override desa (dari panel admin) SEBELUM sync realisasi,
# karena realisasi dihitung berdasarkan angka warga wajib edukasi
# yang sudah ter-update.
sync_desa_overrides()


# ========================================================
# KAJIAN RISIKO BENCANA (tabulasi BPBD, lihat kajian_risiko.py)
# ========================================================
# Tingkat risiko per jenis bencana dibaca LANGSUNG dari file matriks
# di folder data/ (bukan dihitung dari rumus). Kalau file tidak ada /
# strukturnya berubah, MATRIKS_KAJIAN = None dan tab peta menampilkan
# info "data belum tersedia" -- aplikasi lain tetap jalan normal.
MATRIKS_KAJIAN = kajian_risiko.muat_matriks_default(app.root_path)

# Data yang diunggah / diisi manual admin disimpan di Firebase (node
# 'kajian_risiko'), karena folder deploy Vercel read-only. Kalau node itu
# ada, ia MENGGANTIKAN file bawaan di atas; kalau belum pernah ada, file
# bawaan tetap dipakai.
_KAJIAN_VERSI = None           # 'version' data Firebase yang sedang dipakai instance ini
_KAJIAN_DI_FIREBASE = False    # True kalau data aktif sudah tersimpan di Firebase
_KAJIAN_LOCK = threading.Lock()


def _pasang_matriks_kajian(m, dari_firebase):
    global MATRIKS_KAJIAN, _KAJIAN_VERSI, _KAJIAN_DI_FIREBASE
    MATRIKS_KAJIAN = m
    _KAJIAN_DI_FIREBASE = dari_firebase
    _KAJIAN_VERSI = (m.get('meta') or {}).get('version') if dari_firebase else None


def sync_kajian_risiko():
    """Dipanggil sekali saat server start (pola sama seperti sync_desa_overrides)."""
    try:
        data = _fb_get('kajian_risiko')
    except Exception as e:
        print(f"⚠️  Gagal membaca Kajian Risiko dari Firebase, memakai data bawaan: {e}")
        return

    m = kajian_risiko.matriks_dari_dict(data)
    if m is None:
        print("ℹ️  Kajian Risiko: belum ada unggahan admin, memakai file bawaan.")
        return

    _pasang_matriks_kajian(m, True)
    print(
        f"✅ Kajian Risiko dimuat dari Firebase [{m['file']}] "
        f"({len(m['desa'])} desa, {len(m['kec'])} kecamatan)."
    )


def refresh_kajian_jika_berubah():
    """
    Dipanggil setiap peta dibangun ulang (paling sering tiap MAP_CACHE_TTL).
    Kalau admin menyimpan data baru lewat instance server LAIN, versinya di
    Firebase berubah -> data diambil ulang di sini. Biayanya 1 baca kecil
    (hanya angka versi) selama tidak ada perubahan.
    """
    try:
        versi = db.reference('kajian_risiko/meta/version').get()
        if versi is None or versi == _KAJIAN_VERSI:
            return
        m = kajian_risiko.matriks_dari_dict(db.reference('kajian_risiko').get())
        if m is not None:
            with _KAJIAN_LOCK:
                _pasang_matriks_kajian(m, True)
    except Exception as e:  # noqa: BLE001 - peta tetap dibuat dgn data yg ada
        print(f"⚠️  Gagal menyegarkan Kajian Risiko dari Firebase: {e}")


sync_kajian_risiko()


@lru_cache(maxsize=1)
def _js_kajian_risiko():
    """Isi templates/kajian_risiko_map.js (dibaca sekali, dipakai tiap generate peta)."""
    path = os.path.join(app.root_path, "templates", "kajian_risiko_map.js")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def sync_realisasi_edukasi():
    """
    Hitung ulang dari NOL seluruh realisasi edukasi berdasarkan
    SEMUA laporan yang ada di Firebase, lalu isi ke desa_dict &
    kec_dict:
        - Jumlah_Teredukasi_Aktual_Desa / _Kec (jumlah orang, akumulasi)
        - Persen_Realisasi_Edukasi_Desa / _Kec (%, dibatasi maks 100%)

    Dipanggil sekali saat server pertama kali start supaya laporan
    lama yang sudah ada di Firebase ikut terhitung (bukan mulai dari
    0). Untuk update instan setiap ada 1 laporan baru masuk (tanpa
    perlu baca ulang seluruh Firebase), pakai tambah_realisasi_edukasi()
    di bawah -- itu yang dipanggil dari route petugas_lapor().
    """

    for d in desa_dict.values():
        d['Jumlah_Teredukasi_Aktual_Desa'] = 0

    for k in kec_dict.values():
        k['Jumlah_Teredukasi_Aktual_Kec'] = 0

    try:
        laporan_data = _fb_get('laporan_edukasi') or {}
    except Exception as e:
        print(f"⚠️  Gagal sinkronisasi realisasi edukasi dari Firebase: {e}")
        laporan_data = {}

    for laporan_id, info in laporan_data.items():

        if not isinstance(info, dict):
            continue

        key_d = clean_name(info.get('desa', ''))
        jumlah_orang = safe_number(info.get('jumlah_orang_teredukasi', 0))

        if key_d in desa_dict:
            desa_dict[key_d]['Jumlah_Teredukasi_Aktual_Desa'] += jumlah_orang

    # Hitung persentase per desa, sekaligus akumulasikan ke kecamatan induknya
    for key_d, d in desa_dict.items():

        target_desa = safe_number(d.get('Warga_Wajib_Edukasi_Desa', 0))
        aktual_desa = d.get('Jumlah_Teredukasi_Aktual_Desa', 0)

        d['Persen_Realisasi_Edukasi_Desa'] = round(
            min(100, (aktual_desa / target_desa * 100)), 2
        ) if target_desa > 0 else 0.0

        key_k = clean_name(d.get('Kecamatan', ''))

        if key_k in kec_dict:
            kec_dict[key_k]['Jumlah_Teredukasi_Aktual_Kec'] += aktual_desa

    for key_k, k in kec_dict.items():

        target_kec = safe_number(k.get('Total_Warga_Wajib_Edukasi_Kecamatan', 0))
        aktual_kec = k.get('Jumlah_Teredukasi_Aktual_Kec', 0)

        k['Persen_Realisasi_Edukasi_Kec'] = round(
            min(100, (aktual_kec / target_kec * 100)), 2
        ) if target_kec > 0 else 0.0

    print(
        f"✅ Realisasi edukasi tersinkron dari Firebase "
        f"({len(laporan_data)} laporan)."
    )


def tambah_realisasi_edukasi(kecamatan_name, desa_name, jumlah_orang_teredukasi):
    """
    Update INSTAN ke desa_dict & kec_dict yang sedang berjalan di
    memori setiap ada SATU laporan edukasi baru masuk dari petugas
    (tanpa perlu baca ulang seluruh Firebase / restart server), supaya
    halaman /kondisi (peta, popup, chatbot) langsung mencerminkan
    laporan terbaru begitu petugas klik "Kirim Laporan Edukasi".
    """

    key_d = clean_name(desa_name)
    key_k = clean_name(kecamatan_name)

    if key_d in desa_dict:

        d = desa_dict[key_d]
        d['Jumlah_Teredukasi_Aktual_Desa'] = (
            d.get('Jumlah_Teredukasi_Aktual_Desa', 0) + jumlah_orang_teredukasi
        )

        target_desa = safe_number(d.get('Warga_Wajib_Edukasi_Desa', 0))
        d['Persen_Realisasi_Edukasi_Desa'] = round(
            min(100, (d['Jumlah_Teredukasi_Aktual_Desa'] / target_desa * 100)), 2
        ) if target_desa > 0 else 0.0

    if key_k in kec_dict:

        k = kec_dict[key_k]
        k['Jumlah_Teredukasi_Aktual_Kec'] = (
            k.get('Jumlah_Teredukasi_Aktual_Kec', 0) + jumlah_orang_teredukasi
        )

        target_kec = safe_number(k.get('Total_Warga_Wajib_Edukasi_Kecamatan', 0))
        k['Persen_Realisasi_Edukasi_Kec'] = round(
            min(100, (k['Jumlah_Teredukasi_Aktual_Kec'] / target_kec * 100)), 2
        ) if target_kec > 0 else 0.0

    # Laporan baru mengubah warna/tooltip peta. Invalidate cache agar
    # request /kondisi berikutnya membangun versi terbaru.
    invalidate_map_cache()


# Sinkronkan sekali saat server start supaya laporan yang sudah ada
# di Firebase sebelumnya ikut terhitung sejak awal (bukan mulai 0).
sync_realisasi_edukasi()


# ========================================================
# CACHE PETA
# ========================================================
# Peta Folium berukuran besar dan relatif statis. Jangan generate ulang
# untuk setiap request /kondisi. Cache ini berlaku per instance Python
# (sangat cocok untuk warm instance Vercel) dan di-invalidasi saat data
# yang memengaruhi warna/tooltip peta berubah.
MAP_CACHE_TTL = int(os.getenv("MAP_CACHE_TTL", "300"))  # 5 menit
_MAP_CACHE = {"html": None, "created_at": 0.0}
_MAP_CACHE_LOCK = threading.Lock()

def invalidate_map_cache():
    """Paksa peta dibuat ulang pada request /kondisi berikutnya."""
    with _MAP_CACHE_LOCK:
        _MAP_CACHE["html"] = None
        _MAP_CACHE["created_at"] = 0.0


def get_cached_map():
    """Ambil HTML peta dari cache; generate hanya jika cache kedaluwarsa."""
    now = time.time()
    with _MAP_CACHE_LOCK:
        cached = _MAP_CACHE.get("html")
        created_at = float(_MAP_CACHE.get("created_at") or 0)
        if cached is not None and (now - created_at) < MAP_CACHE_TTL:
            return cached

        # Lock sengaja ditahan saat generate agar dua request bersamaan
        # tidak sama-sama menjalankan Folium yang mahal.
        refresh_kajian_jika_berubah()
        html = generate_map()
        _MAP_CACHE["html"] = html
        _MAP_CACHE["created_at"] = time.time()
        return html


# ========================================================
# GEOJSON
# ========================================================
 
# ========================================================
# PEMADATAN KOORDINAT GEOJSON
# ========================================================
# File GeoJSON batas wilayah menyimpan rata-rata ~10 digit desimal per
# koordinat (presisi sub-milimeter -- jauh melebihi kebutuhan peta). Semua
# koordinat itu ikut ditanam ke halaman peta (~6 MB). Saat dibaca, koordinat
# dibulatkan ke 5 desimal (~1 meter; selisih maksimum ~0,55 m, tak terlihat
# bahkan di zoom terdekat) dan titik kembar berurutan akibat pembulatan
# dibuang. File aslinya di static/ TIDAK diubah. Vertex yang dipakai
# bersama dua wilayah bertetangga dibulatkan sama persis, jadi tidak muncul
# celah di antara poligon.
KOORDINAT_DESIMAL = 5


def _bulatkan_koordinat(coords, desimal=KOORDINAT_DESIMAL):
    if not isinstance(coords, list) or not coords:
        return coords

    # daun: satu titik [lon, lat, (z)]
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], desimal), round(coords[1], desimal)] + coords[2:]

    hasil = [_bulatkan_koordinat(c, desimal) for c in coords]

    # daftar titik (ring / garis): buang titik kembar yang berurutan
    pertama = hasil[0]
    if isinstance(pertama, list) and pertama and isinstance(pertama[0], (int, float)):
        tanpa_kembar = [pertama]
        for titik in hasil[1:]:
            if titik[:2] != tanpa_kembar[-1][:2]:
                tanpa_kembar.append(titik)
        # ring poligon minimal 4 titik; kalau jadi kurang, biarkan apa adanya
        if len(tanpa_kembar) >= 4 or len(hasil) < 4:
            hasil = tanpa_kembar

    return hasil


def _padatkan_geojson_features(features):
    for feature in features:
        geom = feature.get("geometry") if isinstance(feature, dict) else None
        if isinstance(geom, dict) and "coordinates" in geom:
            geom["coordinates"] = _bulatkan_koordinat(geom["coordinates"])


@lru_cache(maxsize=1)
def load_local_geojson_files():
 
    folder = os.path.join(
        app.root_path,
        "static",
        "id3217_bandung_barat"
    )
 
    features = []
 
    if not os.path.exists(folder):
 
        print(
            "❌ Folder GeoJSON tidak ditemukan:",
            folder
        )
 
        return {
            "type": "FeatureCollection",
            "features": []
        }
 
 
    print("\n========================================")
    print("MEMBACA GEOJSON DESA")
    print("========================================")
 
 
    for filename in os.listdir(folder):
 
        if not filename.lower().endswith(
            (".geojson", ".json")
        ):
            continue
 
        if "32.17_kecamatan" in filename.lower():
            continue
 
        if "waduk" in filename.lower():
            continue
 
 
        filepath = os.path.join(
            folder,
            filename
        )
 
 
        try:
 
            nama_file = os.path.splitext(
                filename
            )[0]
 
            parts = nama_file.split(
                "_",
                1
            )
 
            nama_kecamatan = (
                parts[1]
                if len(parts) == 2
                else ""
            )
 
            nama_kecamatan = (
                nama_kecamatan
                .replace("-", " ")
                .strip()
            )
 
 
            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as f:
 
                geojson_data = json.load(f)
 
 
            if geojson_data.get(
                "type"
            ) == "FeatureCollection":
 
                file_features = (
                    geojson_data.get(
                        "features",
                        []
                    )
                )
 
            elif geojson_data.get(
                "type"
            ) == "Feature":
 
                file_features = [
                    geojson_data
                ]
 
            else:
 
                continue
 
 
            _padatkan_geojson_features(file_features)

            for feature in file_features:
 
                if "properties" not in feature:
 
                    feature["properties"] = {}
 
 
                feature["properties"][
                    "nm_kecamatan"
                ] = nama_kecamatan
 
 
                features.append(feature)
 
 
        except Exception as e:
 
            print(
                f"❌ Gagal membaca "
                f"{filename}: {e}"
            )
 
 
    return {
        "type": "FeatureCollection",
        "features": features
    }
 
 
# ========================================================
# GET NAMA DESA
# ========================================================
 
def get_nama_desa(properties):
 
    for key in [
        "village",
        "nama_desa",
        "desa",
        "nm_desa",
        "DESA",
        "NAMOBJ"
    ]:
 
        if (
            key in properties
            and properties[key]
        ):
 
            return str(
                properties[key]
            )
 
    return ""
 
 
# ========================================================
# GET NAMA KECAMATAN
# ========================================================
 
def get_nama_kecamatan(properties):
 
    if not properties:
 
        return ""
 
 
    nama = properties.get(
        "nm_kecamatan"
    )
 
    if nama:
 
        return str(
            nama
        ).strip()
 
 
    for key in [
        "kecamatan",
        "Kecamatan",
        "KECAMATAN",
        "nama_kecamatan"
    ]:
 
        if (
            key in properties
            and properties[key]
        ):
 
            return str(
                properties[key]
            ).strip()
 
 
    return ""


# ========================================================
# DATA KOSONG (DESA/KECAMATAN BELUM TERCATAT)
# ========================================================
#
# Membandingkan SELURUH desa yang ada di GeoJSON (batas wilayah
# administratif riil, dari HDX-BPS) dengan desa yang sudah
# tercatat di desa_dict (hasil olahan Excel). Desa yang muncul
# di peta tapi TIDAK ada di desa_dict berarti datanya belum
# diinput / belum tercatat oleh petugas.
# ========================================================

def hitung_data_kosong():
    """
    Return:
        desa_kosong_list: list of dict, tiap desa yang belum
            tercatat -> {"Desa": ..., "Kecamatan": ..., "kec_key": ...}
        kec_kosong_list: list of dict, dikelompokkan per
            kecamatan -> {"Kecamatan": ..., "jumlah_desa_kosong": ...,
            "daftar_desa": [...]}
    """

    geojson_data = load_local_geojson_files()

    desa_kosong_list = []
    seen_keys = set()

    for feature in geojson_data.get("features", []):

        properties = feature.get("properties", {})

        raw_desa = get_nama_desa(properties)
        key_d = clean_name(raw_desa)

        # Lewati fitur tanpa nama / fitur non-desa (mis. waduk)
        if not key_d or key_d == "waduk":
            continue

        # Satu desa bisa punya >1 polygon (multi-part), hindari duplikat
        if key_d in seen_keys:
            continue

        seen_keys.add(key_d)

        if key_d not in desa_dict:

            raw_kec = (
                properties.get("district")
                or get_nama_kecamatan(properties)
                or ""
            )

            desa_kosong_list.append({
                "Desa": raw_desa.strip().title() if raw_desa else "(Tanpa Nama)",
                "Kecamatan": raw_kec.strip().title() if raw_kec else "(Tidak Diketahui)",
                "kec_key": clean_name(raw_kec) or "tidak-diketahui",
            })

    desa_kosong_list.sort(key=lambda d: (d["Kecamatan"], d["Desa"]))

    # --- Kelompokkan per kecamatan ---
    kec_kosong_map = {}

    for d in desa_kosong_list:

        kk = d["kec_key"]

        if kk not in kec_kosong_map:
            kec_kosong_map[kk] = {
                "Kecamatan": d["Kecamatan"],
                "jumlah_desa_kosong": 0,
                "daftar_desa": [],
            }

        kec_kosong_map[kk]["jumlah_desa_kosong"] += 1
        kec_kosong_map[kk]["daftar_desa"].append(d["Desa"])

    kec_kosong_list = sorted(
        kec_kosong_map.values(),
        key=lambda k: k["jumlah_desa_kosong"],
        reverse=True,
    )

    return desa_kosong_list, kec_kosong_list


# Dihitung sekali saat server start. Dibungkus try/except supaya kalau
# ada masalah baca file GeoJSON di environment tertentu (mis. serverless),
# fitur "Data Kosong" saja yang nonaktif -- TIDAK menjatuhkan seluruh
# aplikasi (dashboard, login, peta, dll tetap jalan normal).
try:
    DESA_KOSONG_LIST, KEC_KOSONG_LIST = hitung_data_kosong()

    print("\n========================================")
    print("DATA DESA BELUM TERCATAT")
    print("========================================")
    print(f"Total desa belum tercatat : {len(DESA_KOSONG_LIST)}")
    print(f"Total kecamatan terdampak : {len(KEC_KOSONG_LIST)}")

except Exception as e:
    print(f"❌ Gagal menghitung data kosong (fitur ini akan dinonaktifkan): {e}")
    DESA_KOSONG_LIST, KEC_KOSONG_LIST = [], []


# ========================================================
# WARNA DESA
# ========================================================
 
def get_color_desa(persen):
 
    """
    Menentukan warna berdasarkan Persentase Prioritas Warga
    yang Teredukasi di desa (warga wajib edukasi desa dibagi
    warga terpapar/wajib edukasi kecamatan x 100). Ini metrik
    PRIORITAS/BOBOT wilayah, bukan realisasi edukasi sungguhan
    -- realisasi sungguhan ada di 'Persen_Realisasi_Edukasi_Desa'
    (dihitung dari laporan petugas) dan ditampilkan terpisah di
    popup/tooltip peta.
 
    >= 30%  : Sangat Tinggi
    15-29.9 : Tinggi
    5-14.9  : Sedang
    < 5%    : Rendah
    """
 
    if persen >= 30:
 
        return "#800026"
 
    elif persen >= 15:
 
        return "#BD0026"
 
    elif persen >= 5:
 
        return "#FD8D3C"
 
    else:
 
        return "#FED976"


# ========================================================
# WARNA GRADASI REALISASI EDUKASI (CHOROPLETH DINAMIS)
# ========================================================
#
# Beda dengan get_color_desa() di atas (yang mewarnai berdasarkan
# BOBOT PRIORITAS wilayah -- 4 tingkat warna tetap), fungsi ini
# mewarnai berdasarkan RASIO REALISASI: berapa persen dari warga
# prioritas (warga wajib edukasi) yang SUDAH benar-benar diedukasi,
# yaitu field 'Persen_Realisasi_Edukasi_Desa' / '_Kec' yang sudah
# dihitung live dari akumulasi laporan petugas di Firebase
# (lihat sync_realisasi_edukasi() & tambah_realisasi_edukasi()).
#
# Semakin TINGGI rasio ini (warga teredukasi semakin banyak
# dibanding warga prioritas, atau warga prioritas semakin sedikit
# karena datanya sudah diedukasi) -> warna semakin HIJAU.
# Semakin RENDAH -> warna semakin MERAH.
# Warnanya bergradasi HALUS (interpolasi linear per titik warna),
# bukan cuma beberapa kelas warna tetap, supaya perubahan kecil pun
# (mis. 40% -> 42%) tetap terlihat sedikit berubah di peta.
# ========================================================

_GRADIENT_STOPS_REALISASI = [
    (0,   (215, 25,  28)),   # 0%   -> merah   #d7191c (belum ada yang teredukasi)
    (50,  (255, 255, 191)),  # 50%  -> kuning  #ffffbf (baru separuh jalan)
    (100, (26,  150, 65)),   # 100% -> hijau   #1a9641 (semua warga prioritas sudah teredukasi)
]


def get_color_realisasi(persen):
    """
    Interpolasi warna MERAH -> KUNING -> HIJAU secara linear
    berdasarkan 'persen' (0-100), dipakai untuk mewarnai
    choropleth desa & kecamatan berdasarkan rasio realisasi
    edukasi (teredukasi / prioritas x 100).
    """

    p = safe_number(persen)
    p = max(0.0, min(100.0, p))

    stops = _GRADIENT_STOPS_REALISASI

    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]

        if p0 <= p <= p1:
            t = (p - p0) / (p1 - p0) if p1 > p0 else 0
            r = round(c0[0] + (c1[0] - c0[0]) * t)
            g = round(c0[1] + (c1[1] - c0[1]) * t)
            b = round(c0[2] + (c1[2] - c0[2]) * t)
            return f"#{r:02x}{g:02x}{b:02x}"

    return "#cccccc"


def get_color_prioritas(sisa, vmin, vmax):
    """
    Warna berdasarkan JUMLAH (bukan persentase) warga prioritas yang
    MASIH TERSISA (belum diedukasi) = warga wajib edukasi - warga yang
    sudah diedukasi. Angka ini otomatis berkurang tiap ada laporan
    edukasi baru masuk, karena diambil langsung dari
    Jumlah_Teredukasi_Aktual_Desa/_Kec yang live.

    Posisi wilayah ini dibandingkan RELATIF terhadap wilayah lain
    (vmin..vmax = sisa prioritas paling sedikit s/d paling banyak di
    seluruh desa/kecamatan saat itu), supaya wilayah dengan penduduk
    jauh lebih besar/kecil tetap adil dibandingkan.

    Sisa BANYAK -> MERAH (prioritas tinggi, masih perlu banyak edukasi)
    Sisa SEDIKIT -> HIJAU (prioritas rendah, sudah hampir tuntas)
    """

    sisa = safe_number(sisa)

    if vmax <= vmin:
        posisi = 0.0
    else:
        posisi = (sisa - vmin) / (vmax - vmin)

    posisi = max(0.0, min(1.0, posisi))

    # posisi 1 (sisa terbanyak) harus jadi merah -> persen 0
    # posisi 0 (sisa tersedikit) harus jadi hijau -> persen 100
    return get_color_realisasi(100 - (posisi * 100))


def hitung_skor_prioritas(data, tingkat="desa"):
    if tingkat == "desa":
        persen_rentan = safe_number(
            data.get("Persen_Rentan_Desa", 0)
        )

        persen_prioritas = safe_number(
            data.get("Persen_Teredukasi_Desa", 0)
        )

        persen_realisasi = safe_number(
            data.get("Persen_Realisasi_Edukasi_Desa", 0)
        )

    else:
        persen_rentan = safe_number(
            data.get("Persen_Rentan_Kec", 0)
        )

        persen_prioritas = safe_number(
            data.get("Persen_Teredukasi_Kecamatan", 0)
        )

        persen_realisasi = safe_number(
            data.get("Persen_Realisasi_Edukasi_Kec", 0)
        )

    # Semakin rendah realisasi, semakin tinggi kebutuhan
    kebutuhan_edukasi = 100 - persen_realisasi

    skor = (
        (0.40 * persen_rentan)
        + (0.30 * persen_prioritas)
        + (0.30 * kebutuhan_edukasi)
    )

    return round(skor, 2)


def cari_prioritas_daerah(pesan):

    kata_prioritas = [
        "daerah yang dijadikan prioritas",
        "daerah prioritas",
        "wilayah prioritas",
        "desa prioritas",
        "kecamatan prioritas",
        "yang menjadi prioritas",
        "prioritas saat ini",
        "daerah mana yang diprioritaskan",
        "wilayah mana yang diprioritaskan"
    ]

    if not any(kata in pesan for kata in kata_prioritas):
        return None

    hasil = []

    # =========================
    # DESA
    # =========================
    for d in desa_dict.values():

        skor = hitung_skor_prioritas(d, "desa")

        hasil.append({
            "nama": d.get("Desa", "-"),
            "kecamatan": d.get("Kecamatan", "-"),
            "tingkat": "Desa",
            "rentan": safe_number(
                d.get("Persen_Rentan_Desa", 0)
            ),
            "prioritas": safe_number(
                d.get("Persen_Teredukasi_Desa", 0)
            ),
            "realisasi": safe_number(
                d.get("Persen_Realisasi_Edukasi_Desa", 0)
            ),
            "skor": skor
        })

    # Urutkan dari skor tertinggi
    hasil.sort(
        key=lambda x: x["skor"],
        reverse=True
    )

    # Ambil 5 wilayah paling prioritas
    top = hasil[:5]

    if not top:
        return (
            "Belum tersedia data yang cukup untuk "
            "menentukan wilayah prioritas."
        )

    reply = (
        "### 🎯 Daerah yang Menjadi Prioritas Saat Ini\n\n"
        "Penentuan prioritas mempertimbangkan:\n"
        "- **40%** persentase kelompok rentan\n"
        "- **30%** persentase prioritas edukasi\n"
        "- **30%** kebutuhan edukasi yang belum tercapai\n\n"
        "**5 daerah dengan skor prioritas tertinggi:**\n\n"
    )

    for i, d in enumerate(top, 1):

        reply += (
            f"**{i}. Desa {d['nama']}** "
            f"(Kec. {d['kecamatan']})\n"
            f"- Persentase rentan: **{d['rentan']:.2f}%**\n"
            f"- Persentase prioritas: **{d['prioritas']:.2f}%**\n"
            f"- Realisasi edukasi: **{d['realisasi']:.2f}%**\n"
            f"- **Skor prioritas: {d['skor']:.2f}**\n\n"
        )

    reply += (
        "Semakin tinggi skor, semakin tinggi kebutuhan "
        "wilayah tersebut untuk diprioritaskan dalam edukasi "
        "kebencanaan."
    )

    return reply

def cari_skor_prioritas_wilayah(pesan):
    """
    Menjawab pertanyaan spesifik mengenai skor prioritas
    suatu desa atau kecamatan.
    """

    kata_skor = [
        "skor prioritas",
        "nilai prioritas",
        "skor prioritasnya",
        "nilai prioritasnya"
    ]

    if not any(kata in pesan for kata in kata_skor):
        return None

    # ==========================================
    # CARI DESA
    # ==========================================

    for key, d in desa_dict.items():

        nama_desa = str(d.get("Desa", "")).lower()

        if key in pesan or nama_desa in pesan:

            skor = hitung_skor_prioritas(d, "desa")

            persen_rentan = safe_number(
                d.get("Persen_Rentan_Desa", 0)
            )

            persen_prioritas = safe_number(
                d.get("Persen_Teredukasi_Desa", 0)
            )

            persen_realisasi = safe_number(
                d.get("Persen_Realisasi_Edukasi_Desa", 0)
            )

            sisa_target = 100 - persen_realisasi

            return (
                f"### 🎯 Skor Prioritas Desa {d.get('Desa')}\n\n"
                f"**Skor Prioritas: {skor:.2f}**\n\n"
                f"Perhitungannya berdasarkan:\n"
                f"- Persentase kelompok rentan: **{persen_rentan:.2f}%**\n"
                f"- Persentase prioritas edukasi: **{persen_prioritas:.2f}%**\n"
                f"- Sisa target edukasi: **{sisa_target:.2f}%** "
                f"(realisasi **{persen_realisasi:.2f}%**)\n\n"
                f"Rumus:\n"
                f"**(40% × rentan) + "
                f"(30% × prioritas) + "
                f"(30% × sisa target edukasi)**"
            )

    # ==========================================
    # CARI KECAMATAN
    # ==========================================

    for key, k in kec_dict.items():

        nama_kec = str(k.get("Kecamatan", "")).lower()

        if key in pesan or nama_kec in pesan:

            skor = hitung_skor_prioritas(k, "kecamatan")

            persen_rentan = safe_number(
                k.get("Persen_Rentan_Kec", 0)
            )

            persen_prioritas = safe_number(
                k.get("Persen_Teredukasi_Kecamatan", 0)
            )

            persen_realisasi = safe_number(
                k.get("Persen_Realisasi_Edukasi_Kec", 0)
            )

            sisa_target = 100 - persen_realisasi

            return (
                f"### 🎯 Skor Prioritas Kecamatan {k.get('Kecamatan')}\n\n"
                f"**Skor Prioritas: {skor:.2f}**\n\n"
                f"Perhitungannya berdasarkan:\n"
                f"- Persentase kelompok rentan: **{persen_rentan:.2f}%**\n"
                f"- Persentase prioritas edukasi: **{persen_prioritas:.2f}%**\n"
                f"- Sisa target edukasi: **{sisa_target:.2f}%** "
                f"(realisasi **{persen_realisasi:.2f}%**)\n\n"
                f"Rumus:\n"
                f"**(40% × rentan) + "
                f"(30% × prioritas) + "
                f"(30% × sisa target edukasi)**"
            )

    return (
        "Saya belum menemukan nama desa atau kecamatan "
        "yang dimaksud. Coba tuliskan nama wilayahnya."
    )
 
# ========================================================
# GENERATE MAP
# ========================================================
 
def generate_map():

    # ====================================================
    # 0. SIAPKAN PETA WARNA 2 MODE: "realisasi" & "prioritas"
    # ====================================================
    #
    # Dihitung SEKALI di awal (bukan di dalam style_desa/
    # style_kecamatan tiap fitur) supaya normalisasi min-max untuk
    # mode "prioritas" konsisten dibandingkan seluruh desa/kecamatan,
    # dan supaya nilainya bisa dikirim ke JS (client-side) sebagai
    # lookup table -- jadi tombol toggle mode di peta bisa langsung
    # ganti warna TANPA reload halaman/panggil server lagi.
    #
    #   - mode "realisasi": dasar Persen_Realisasi_Edukasi_Desa/_Kec
    #     (rasio %, sudah ada dari fitur sebelumnya)
    #   - mode "prioritas": dasar JUMLAH warga prioritas yang MASIH
    #     tersisa (warga wajib edukasi - yang sudah diedukasi), jadi
    #     otomatis ikut berkurang/update setiap ada laporan edukasi
    #     baru masuk, karena sumbernya field yang sama & live.
    # ====================================================

    sisa_prioritas_desa = {}
    for key_d, d in desa_dict.items():
        target = safe_number(d.get('Warga_Wajib_Edukasi_Desa', 0))
        aktual = safe_number(d.get('Jumlah_Teredukasi_Aktual_Desa', 0))
        sisa_prioritas_desa[key_d] = max(0, target - aktual)

    sisa_prioritas_kec = {}
    for key_k, k in kec_dict.items():
        target = safe_number(k.get('Total_Warga_Wajib_Edukasi_Kecamatan', 0))
        aktual = safe_number(k.get('Jumlah_Teredukasi_Aktual_Kec', 0))
        sisa_prioritas_kec[key_k] = max(0, target - aktual)

    vmin_desa = min(sisa_prioritas_desa.values()) if sisa_prioritas_desa else 0
    vmax_desa = max(sisa_prioritas_desa.values()) if sisa_prioritas_desa else 0
    vmin_kec = min(sisa_prioritas_kec.values()) if sisa_prioritas_kec else 0
    vmax_kec = max(sisa_prioritas_kec.values()) if sisa_prioritas_kec else 0

    color_map_desa = {}
    for key_d, d in desa_dict.items():
        color_map_desa[key_d] = {
            "realisasi": get_color_realisasi(
                d.get('Persen_Realisasi_Edukasi_Desa', 0)
            ),
            "prioritas": get_color_prioritas(
                sisa_prioritas_desa[key_d], vmin_desa, vmax_desa
            ),
        }

    color_map_kec = {}
    for key_k, k in kec_dict.items():
        color_map_kec[key_k] = {
            "realisasi": get_color_realisasi(
                k.get('Persen_Realisasi_Edukasi_Kec', 0)
            ),
            "prioritas": get_color_prioritas(
                sisa_prioritas_kec[key_k], vmin_kec, vmax_kec
            ),
        }

    # Skor Destana desa = persentase indikator aktif yang terpenuhi.
    # Skor Destana kecamatan = rata-rata skor seluruh desa di kecamatan.
    destana_indicators, threshold_now, destana_checklists = load_destana_snapshot()
    destana_score_desa = {}
    destana_status_desa = {}
    destana_missing_desa = {}
    destana_checked_desa = {}
    destana_scores_by_kec = {}
    destana_counts_by_kec = {}

    for key_d, d in desa_dict.items():
        rec = get_destana_record(
            d.get('Kecamatan', ''), d.get('Desa', ''), destana_indicators,
            destana_checklists, threshold_now,
        )
        score = round(float(rec.get('persen', 0)), 2)
        destana_score_desa[key_d] = score
        destana_status_desa[key_d] = rec.get('status', 'Belum memenuhi')

        # Daftar nama indikator Destana yang SUDAH & BELUM dicentang
        # untuk desa ini -- dipakai di tooltip mode "destana" supaya
        # petugas langsung tahu apa saja yang sudah & masih kurang,
        # tanpa perlu buka halaman Destana desa itu satu-satu.
        checked = rec.get('checked', {})
        active_indicators = [
            (iid, info) for iid, info in destana_indicators.items()
            if isinstance(info, dict) and info.get('aktif', True)
        ]
        destana_missing_desa[key_d] = [
            info.get('nama', '') for iid, info in active_indicators
            if not checked.get(iid)
        ]
        destana_checked_desa[key_d] = [
            info.get('nama', '') for iid, info in active_indicators
            if checked.get(iid)
        ]

        key_k = clean_name(d.get('Kecamatan', ''))
        if key_k:
            destana_scores_by_kec.setdefault(key_k, []).append(score)
            destana_counts_by_kec.setdefault(key_k, {'total': 0, 'destana': 0})
            destana_counts_by_kec[key_k]['total'] += 1
            if rec.get('status') == 'Destana':
                destana_counts_by_kec[key_k]['destana'] += 1

    destana_score_kec = {}
    destana_status_kec = {}
    for key_k in kec_dict:
        scores = destana_scores_by_kec.get(key_k, [])
        avg = sum(scores) / len(scores) if scores else 0
        destana_score_kec[key_k] = round(avg, 2)
        destana_status_kec[key_k] = 'Destana' if scores and avg >= threshold_now else 'Belum memenuhi'

    color_map_destana_desa = {k: get_color_realisasi(v) for k, v in destana_score_desa.items()}
    color_map_destana_kec = {k: get_color_realisasi(v) for k, v in destana_score_kec.items()}

    # Tab "Kajian Risiko Bencana": warna, tooltip & legenda-nya dibangun
    # di sisi JS dari data tabulasi (lihat kajian_risiko.py &
    # templates/kajian_risiko_map.js) -- datanya disiapkan di bagian
    # "3. LEGEND" di bawah, setelah GeoJSON desa & kecamatan terbaca.

    # Tooltip peta kini DIPISAH per mode/tab (realisasi, prioritas,
    # destana; tab kajian risiko dibangun di JS) supaya tiap tab cuma menampilkan info
    # yang relevan buat tab itu -- bukan 1 tooltip raksasa gabungan
    # semua info seperti sebelumnya. Diisi di dalam loop kecamatan &
    # desa di bawah, lalu dikirim ke JS (lihat __TOOLTIP_MAP_*_JSON__)
    # supaya applyColorMode() bisa ganti isi tooltip TANPA reload.
    tooltip_map_kec = {}
    tooltip_map_desa = {}

    def _bullet_list_indikator(items, kosong_teks):
        """
        Ubah list nama indikator Destana jadi daftar bullet HTML,
        dipakai di tooltip mode "destana" -- baik untuk daftar yang
        SUDAH dicentang maupun yang BELUM/kurang.
        """
        if not items:
            return f"<br><span style=\"font-size:10px;color:#7f8c8d;\"><i>{kosong_teks}</i></span>"
        tampil = items[:4]
        html = "".join(f"<br>&nbsp;&nbsp;• {item}" for item in tampil)
        if len(items) > 4:
            html += f"<br>&nbsp;&nbsp;<i>+{len(items) - 4} lainnya</i>"
        return html

    m = folium.Map(
        location=[
            -6.8452,
            107.5023
        ],
        zoom_start=11,
        tiles="OpenStreetMap"
    )
 
 
    # ====================================================
    # 1. LAYER KECAMATAN
    # ====================================================
 
    kec_file = os.path.join(
        app.root_path,
        "static",
        "id3217_bandung_barat",
        "32.17_kecamatan.geojson"
    )
 
 
    if not os.path.exists(kec_file):
 
        kec_file = os.path.join(
            app.root_path,
            "static",
            "32.17_kecamatan.geojson"
        )
 
 
    if os.path.exists(kec_file):
 
        with open(
            kec_file,
            "r",
            encoding="utf-8"
        ) as f:
 
            kec_geojson = json.load(f)

        _padatkan_geojson_features(kec_geojson.get("features", []))
 
 
        def style_kecamatan(feature):
 
            properties = feature.get(
                "properties",
                {}
            )
 
            raw_kec = properties.get(
                "nm_kecamatan",
                ""
            )
 
 
            if not raw_kec:
 
                raw_kec = get_nama_kecamatan(
                    properties
                )
 
 
            raw_kec = str(
                raw_kec
            ).strip()
 
 
            key_k = clean_name(
                raw_kec
            )
 
 
            fill_color = "#cccccc"


            if key_k in color_map_kec:

                # ---------------------------------------------
                # WARNA CHOROPLETH DINAMIS -- 2 MODE
                # ---------------------------------------------
                # Nilai awal dirender pakai mode "realisasi" (rasio
                # warga teredukasi / prioritas). Mode "prioritas"
                # (berdasar JUMLAH sisa warga prioritas) tersedia
                # sebagai data JS (colorMapKec) dan diterapkan lewat
                # tombol toggle di peta -- lihat applyColorMode() di
                # interactive_script, tanpa perlu render ulang.
                # ---------------------------------------------

                fill_color = color_map_kec[key_k]["realisasi"]


            return {
 
                "fillColor":
                    fill_color,
 
                "color":
                    "#0f172a",
 
                "weight":
                    2,
 
                "fillOpacity":
                    0.85,
 
                "className":
                    f"layer-kecamatan "
                    f"kec-path-{key_k}"
            }
 
 
        for feature in kec_geojson[
            "features"
        ]:
 
            raw_kec = feature.get(
                "properties",
                {}
            ).get(
                "nm_kecamatan",
                ""
            )
 
 
            if not raw_kec:
 
                raw_kec = get_nama_kecamatan(
                    feature.get(
                        "properties",
                        {}
                    )
                )
 
 
            raw_kec = str(
                raw_kec
            ).strip()
 
 
            key_k = clean_name(
                raw_kec
            )
 
 
            if key_k in kec_dict:
 
                k = kec_dict[key_k]

                skor_destana_kec = destana_score_kec.get(key_k, 0)
                status_destana_kec = destana_status_kec.get(key_k, 'Belum memenuhi')
                warna_destana_kec = '#16a34a' if status_destana_kec == 'Destana' else '#dc2626'
                label_destana_kec = '✅ Destana' if status_destana_kec == 'Destana' else '⚠️ Belum memenuhi'

 
 
                # ---------------------------------------------
                # TOOLTIP DIPISAH PER MODE/TAB -- tiap mode cuma
                # menampilkan info yang relevan buat mode itu saja,
                # supaya gak jadi 1 tooltip raksasa gabungan semua
                # info kayak sebelumnya. Isinya sesuai permintaan:
                #   - realisasi        : warga terpapar & sudah diedukasi
                #   - prioritas        : warga terpapar, kelompok rentan,
                #                        % prioritas warga teredukasi
                #   - destana          : warga terpapar, skor destana,
                #                        daftar indikator yang kurang
                #   - kajian risiko    : dibangun di JS dari tabulasi
                #                        (templates/kajian_risiko_map.js)
                # ---------------------------------------------

                _kec_terpapar = int(k.get('Terpapar_Kecamatan', 0))
                _kec_aktual = int(k.get('Jumlah_Teredukasi_Aktual_Kec', 0))
                _kec_persen_realisasi = k.get('Persen_Realisasi_Edukasi_Kec', 0)
                _kec_total_rentan = int(k.get('Total_Rentan_Kec', 0))
                _kec_persen_prioritas = k['Persen_Teredukasi_Kecamatan']
                _kec_destana_counts = destana_counts_by_kec.get(key_k, {'total': 0, 'destana': 0})

                _tt_head_kec = (
                    f"<b style=\"font-size:13px;color:#2c3e50;\">"
                    f"KEC. {k['Kecamatan'].upper()}</b>"
                    f"<hr style=\"margin:4px 0;border:0;border-top:1px solid #ccc;\">"
                )
                _tt_wrap_open = (
                    "<div style=\"font-family:Arial, sans-serif; "
                    "min-width:190px; padding:4px;\">"
                )
                _tt_wrap_close = "</div>"

                tooltip_map_kec[key_k] = {}

                tooltip_map_kec[key_k]['realisasi'] = (
                    _tt_wrap_open + _tt_head_kec +
                    f"👥 Warga Terpapar: <b>{_kec_terpapar:,}</b> jiwa<br>"
                    f"✅ Sudah Diedukasi: <b>{_kec_aktual:,}</b> jiwa "
                    f"<span style=\"font-size:10px;color:#16a34a;\">"
                    f"({_kec_persen_realisasi:.2f}%)</span>"
                    + _tt_wrap_close
                )

                tooltip_map_kec[key_k]['prioritas'] = (
                    _tt_wrap_open + _tt_head_kec +
                    f"👥 Warga Terpapar: <b>{_kec_terpapar:,}</b> jiwa<br>"
                    f"<u>Detail Kelompok Rentan:</u><br>"
                    f"&nbsp;&nbsp;• Balita/Lansia: <b>{int(k.get('Rentan_BalitaLansia_Kec', 0)):,}</b><br>"
                    f"&nbsp;&nbsp;• Disabilitas: <b>{int(k.get('Rentan_Disabilitas_Kec', 0)):,}</b><br>"
                    f"&nbsp;&nbsp;• Ibu Hamil: <b>{int(k.get('Rentan_IbuHamil_Kec', 0)):,}</b><br>"
                    f"&nbsp;&nbsp;<i>Total: {_kec_total_rentan:,} jiwa</i><br>"
                    f"🎯 Prioritas Warga Teredukasi: "
                    f"<b style=\"color:#d35400;\">{_kec_persen_prioritas:.2f}%</b><br>"
                    # Keterangan kecil (permintaan BPBD): jelaskan basis
                    # hitung persentase. Level kecamatan = porsi satu
                    # kecamatan terhadap total seluruh KBB.
                    f"<span style=\"font-size:10px;color:#7f8c8d;\">"
                    f"<i>Persentase satu kecamatan dari total seluruh KBB</i></span>"
                    + _tt_wrap_close
                )

                tooltip_map_kec[key_k]['destana'] = (
                    _tt_wrap_open + _tt_head_kec +
                    f"👥 Warga Terpapar: <b>{_kec_terpapar:,}</b> jiwa<br>"
                    f"🛡️ Skor Destana (rata-rata): <b style=\"color:{warna_destana_kec};\">"
                    f"{skor_destana_kec:.2f}%</b> "
                    f"<span style=\"font-size:10px;color:{warna_destana_kec};\">"
                    f"({label_destana_kec})</span><br>"
                    f"<span style=\"font-size:10px;\">"
                    f"{_kec_destana_counts.get('destana', 0)} dari "
                    f"{_kec_destana_counts.get('total', 0)} desa sudah berstatus Destana"
                    f"</span><br>"
                    f"<span style=\"font-size:10px;color:#7f8c8d;\">"
                    f"<i>Skor per kecamatan = rata-rata skor desa. Untuk detail "
                    f"indikator yang sudah/belum dicentang, hover ke masing-masing "
                    f"desa.</i></span>"
                    + _tt_wrap_close
                )

                tooltip_html = tooltip_map_kec[key_k]['realisasi']
 
            else:
 
                tooltip_html = (
                    f"<b>Kecamatan: "
                    f"{raw_kec}</b>"
                )
 
 
            folium.GeoJson(
 
                feature,
 
                style_function=
                    style_kecamatan,
 
                tooltip=folium.Tooltip(
                    tooltip_html
                )
 
            ).add_to(m)
 
 
    # ====================================================
    # 2. LAYER DESA
    # ====================================================
 
    desa_geojson = (
        load_local_geojson_files()
    )

    # ----------------------------------------------------
    # DATA KAJIAN RISIKO BENCANA (dari tabulasi BPBD)
    # ----------------------------------------------------
    # Dicocokkan ke fitur GeoJSON dengan kunci GABUNGAN
    # kecamatan|desa (ada nama desa kembar antar kecamatan).
    # Dipakai untuk: popup desa (ringkasan semua bencana) &
    # JS peta (warna/tooltip/legenda tab Kajian Risiko Bencana).
    # ----------------------------------------------------

    _kec_fitur_kajian = (
        kec_geojson.get("features", [])
        if os.path.exists(kec_file)
        else []
    )

    kajian_data_peta = kajian_risiko.siapkan_data_peta(
        MATRIKS_KAJIAN,
        desa_geojson["features"],
        _kec_fitur_kajian,
        get_nama_desa
    )

    if kajian_data_peta["tersedia"]:
        _lap = kajian_data_peta["laporan"]
        if _lap["matriks_tak_terpakai"]:
            print(
                "⚠️  Kajian Risiko: desa di matriks yang TIDAK ketemu "
                "di peta: " + "; ".join(_lap["matriks_tak_terpakai"])
            )
        if _lap["desa_tanpa_data"]:
            print(
                "⚠️  Kajian Risiko: desa di peta yang TIDAK ada di "
                "matriks: " + "; ".join(_lap["desa_tanpa_data"])
            )
 
 
    def style_desa(feature):
 
        raw_desa = get_nama_desa(
            feature["properties"]
        )
 
        key_d = clean_name(
            raw_desa
        )
 
 
        # DEFAULT
        fill_color = "#cccccc"


        if key_d in color_map_desa:

            # ---------------------------------------------
            # WARNA CHOROPLETH DINAMIS -- 2 MODE
            # ---------------------------------------------
            # Nilai awal dirender pakai mode "realisasi" (rasio warga
            # teredukasi / prioritas). Mode "prioritas" (berdasar
            # JUMLAH sisa warga prioritas) tersedia sebagai data JS
            # (colorMapDesa) dan diterapkan lewat tombol toggle di
            # peta -- lihat applyColorMode() di interactive_script,
            # tanpa perlu render ulang dari server.
            # ---------------------------------------------

            fill_color = color_map_desa[key_d]["realisasi"]


        # ---------------------------------------------
        # KECAMATAN INDUK DESA
        # ---------------------------------------------

        raw_kec_parent = ""


        if key_d in desa_dict:

            raw_kec_parent = clean_name(
                desa_dict[key_d].get(
                    "Kecamatan",
                    ""
                )
            )

        else:

            raw_kec_parent = clean_name(
                get_nama_kecamatan(
                    feature["properties"]
                )
            )


        return {

            "fillColor":
                fill_color,

            "color":
                "#ffffff",

            "weight":
                1.5,

            "fillOpacity":
                0.85,

            "className":
                f"layer-desa "
                f"desa-parent-{raw_kec_parent} "
                f"desa-path-{key_d}"
        }
 
 
    for feature in desa_geojson[
        "features"
    ]:
 
        raw_desa = get_nama_desa(
            feature["properties"]
        )
 
        key_d = clean_name(
            raw_desa
        )
 
 
        if key_d == "waduk":
 
            continue
 
 
        if key_d in desa_dict:
 
            d = desa_dict[key_d]
 
 
            persen_teredukasi = safe_number(
                d.get(
                    "Persen_Teredukasi_Desa",
                    0
                )
            )
 
            persen_rentan_desa = safe_number(
                d.get(
                    "Persen_Rentan_Desa",
                    0
                )
            )
 
            kategori_rentan_desa = d.get(
                "Kategori_Rentan_Desa",
                "-"
            )

            skor_destana_desa = destana_score_desa.get(key_d, 0)
            status_destana_desa = destana_status_desa.get(key_d, 'Belum memenuhi')
            warna_destana_desa = '#16a34a' if status_destana_desa == 'Destana' else '#dc2626'
            label_destana_desa = '✅ Destana' if status_destana_desa == 'Destana' else '⚠️ Belum memenuhi'

            # --- Kajian Risiko Bencana (ringkasan semua jenis bencana) ---
            # Dari tabulasi BPBD, bukan rumus. Kunci pakai kecamatan
            # dari GeoJSON (bukan d['Kecamatan']) supaya desa kembar
            # antar kecamatan tidak tertukar.
            _key_kajian_desa = (
                kajian_risiko.key_js(feature["properties"].get("nm_kecamatan", ""))
                + "|" + kajian_risiko.key_js(raw_desa)
            )
            popup_kajian_html = kajian_risiko.popup_ringkasan_html(
                kajian_data_peta["desa"].get(_key_kajian_desa, {}).get("h", {})
            )
 
 
            popup_html = f"""
            <div style="
                font-family: Arial, sans-serif;
                min-width: 280px;
                font-size: 12px;
            ">
 
                <div style="
                    background-color: #2c3e50;
                    color: white;
                    padding: 10px;
                    border-radius: 4px 4px 0 0;
                    text-align: center;
                ">
 
                    <h3 style="
                        margin: 0;
                        font-size: 15px;
                    ">
                        DESA {str(
                            d.get('Desa', '')
                        ).upper()}
                    </h3>
 
                    <span style="
                        font-size: 11px;
                    ">
                        KECAMATAN {str(
                            d.get('Kecamatan', '')
                        ).upper()}
                    </span>
 
                </div>
 
 
                <div style="
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-top: none;
                    background-color: #f8f9fa;
                ">
 
                    <table style="
                        width: 100%;
                        font-size: 12px;
                        color: #34495e;
                    ">
 
                        <tr>
 
                            <td>
                                Warga Wajib Diedukasi
                            </td>
 
                            <td style="
                                text-align: right;
                            ">
                                <b>
                                    {int(
                                        d.get(
                                            'Warga_Wajib_Edukasi_Desa',
                                            0
                                        )
                                    ):,}
                                </b>
                                jiwa
                            </td>
 
                        </tr>
 
 
                        <tr>
 
                            <td colspan="2">
                                <hr style="
                                    margin: 4px 0;
                                ">
                            </td>
 
                        </tr>
 
 
                        <tr>
 
                            <td colspan="2"
                                style="padding-top:2px;">
                                <b>Detail Kelompok Rentan</b>
                            </td>
 
                        </tr>
 
                        <tr>
 
                            <td>
                                &nbsp;&nbsp;• Balita/Lansia
                            </td>
 
                            <td style="text-align: right;">
                                {int(
                                    d.get(
                                        'Rentan_BalitaLansia_Desa',
                                        0
                                    )
                                ):,}
                            </td>
 
                        </tr>
 
                        <tr>
 
                            <td>
                                &nbsp;&nbsp;• Warga Miskin
                            </td>
 
                            <td style="text-align: right;">
                                {int(
                                    d.get(
                                        'Rentan_Miskin_Desa',
                                        0
                                    )
                                ):,}
                            </td>
 
                        </tr>
 
                        <tr>
 
                            <td>
                                &nbsp;&nbsp;• Disabilitas
                            </td>
 
                            <td style="text-align: right;">
                                {int(
                                    d.get(
                                        'Rentan_Disabilitas_Desa',
                                        0
                                    )
                                ):,}
                            </td>
 
                        </tr>
 
                        <tr>
 
                            <td>
                                Persentase Rentan
                            </td>
 
                            <td style="
                                text-align: right;
                            ">
                                <b>
                                    {persen_rentan_desa:.2f}%
                                </b>
                                ({kategori_rentan_desa})
                            </td>
 
                        </tr>
 
 
                        <tr>
 
                            <td colspan="2">
                                <hr style="
                                    margin: 4px 0;
                                ">
                            </td>
 
                        </tr>
 
 
                        <tr>
 
                            <td>
                                Warga Terpapar Kecamatan
                            </td>
 
                            <td style="
                                text-align: right;
                            ">
                                <b>
                                    {int(
                                        d.get(
                                            'Terpapar_Kecamatan',
                                            0
                                        )
                                    ):,}
                                </b>
                                jiwa
                            </td>
 
                        </tr>
 
 
                        <tr>
 
                            <td colspan="2">
                                <hr style="
                                    margin: 4px 0;
                                ">
                            </td>
 
                        </tr>
 
 
                        <tr style="
                            background-color: #e9ecef;
                        ">
 
                            <td style="
                                padding: 6px;
                            ">
 
                                <b>
                                    Persentase Prioritas Warga yang Teredukasi (Kecamatan)
                                </b>
 
                            </td>
 
                            <td style="
                                text-align: right;
                                padding: 6px;
                                color: #c0392b;
                            ">
 
                                <b>
                                    {persen_teredukasi:.2f}%
                                </b>
 
                            </td>
 
                        </tr>

                        <tr style="
                            background-color: #d1fae5;
                        ">

                            <td style="
                                padding: 6px;
                            ">

                                <b>
                                    Persentase Realisasi Edukasi (Real)
                                </b>
                                <br>
                                <span style="font-size: 10px; color: #555;">
                                    dari laporan petugas
                                </span>

                            </td>

                            <td style="
                                text-align: right;
                                padding: 6px;
                                color: #15803d;
                            ">

                                <b>
                                    {d.get('Persen_Realisasi_Edukasi_Desa', 0):.2f}%
                                </b>
                                <br>
                                <span style="font-size: 10px;">
                                    ({int(d.get('Jumlah_Teredukasi_Aktual_Desa', 0)):,} orang)
                                </span>

                            </td>

                        </tr>

                        <tr style="
                            background-color: #f1f5f9;
                        ">

                            <td style="
                                padding: 6px;
                            ">

                                <b>
                                    🛡️ Skor Destana
                                </b>
                                <br>
                                <span style="font-size: 10px; color: {warna_destana_desa};">
                                    {label_destana_desa}
                                </span>

                            </td>

                            <td style="
                                text-align: right;
                                padding: 6px;
                                color: {warna_destana_desa};
                            ">

                                <b>
                                    {skor_destana_desa:.2f}%
                                </b>

                            </td>

                        </tr>

                        <tr style="
                            background-color: #fef2f2;
                        ">

                            <td colspan="2" style="padding: 6px;">

                                <b>
                                    🗺️ Kajian Risiko Bencana
                                </b>
                                <br>
                                <span style="font-size: 10px; color: #7f8c8d;">
                                    Tingkat risiko per jenis bencana (dari tabulasi BPBD)
                                </span>
                                <br>
                                {popup_kajian_html}

                            </td>

                        </tr>
 
                    </table>
 
 
                    <div style="
                        margin-top: 8px;
                        font-size: 10px;
                        color: #7f8c8d;
                    ">
 
                        Rumus:
 
                        <br>
 
                        Warga Wajib Diedukasi Desa ÷
                        Warga Terpapar Kecamatan × 100%
 
                    </div>
 
                </div>
 
            </div>
            """
 
 
            # ---------------------------------------------
            # TOOLTIP DIPISAH PER MODE/TAB (sama seperti kecamatan
            # di atas) -- lihat komentar di blok kecamatan untuk
            # penjelasan lengkap kenapa dipisah.
            # ---------------------------------------------

            _desa_terpapar = int(d.get('Warga_Wajib_Edukasi_Desa', 0))
            _desa_aktual = int(d.get('Jumlah_Teredukasi_Aktual_Desa', 0))
            _desa_persen_realisasi = d.get('Persen_Realisasi_Edukasi_Desa', 0)
            _desa_total_rentan = int(d.get('Total_Rentan_Desa', 0))
            _desa_nama = str(d.get('Desa', '')).upper()

            _list_kurang_html = _bullet_list_indikator(
                destana_missing_desa.get(key_d, []),
                'Semua indikator terpenuhi ✅'
            )
            _list_dicentang_html = _bullet_list_indikator(
                destana_checked_desa.get(key_d, []),
                'Belum ada yang dicentang sama sekali'
            )

            _tt_head_desa = (
                f"<b>DESA {_desa_nama}</b>"
                f"<hr style=\"margin:4px 0;border:0;border-top:1px solid #ccc;\">"
            )
            _tt_wrap_open_d = (
                "<div style=\"font-family:Arial, sans-serif; "
                "min-width:170px; padding:4px;\">"
            )
            _tt_wrap_close_d = "</div>"

            tooltip_map_desa[key_d] = {}

            tooltip_map_desa[key_d]['realisasi'] = (
                _tt_wrap_open_d + _tt_head_desa +
                f"👥 Warga Terpapar: <b>{_desa_terpapar:,}</b> jiwa<br>"
                f"✅ Sudah Diedukasi: <b>{_desa_aktual:,}</b> jiwa "
                f"<span style=\"font-size:10px;color:#16a34a;\">"
                f"({_desa_persen_realisasi:.2f}%)</span>"
                + _tt_wrap_close_d
            )

            tooltip_map_desa[key_d]['prioritas'] = (
                _tt_wrap_open_d + _tt_head_desa +
                f"👥 Warga Terpapar: <b>{_desa_terpapar:,}</b> jiwa<br>"
                f"<u>Detail Kelompok Rentan:</u><br>"
                f"&nbsp;&nbsp;• Balita/Lansia: <b>{int(d.get('Rentan_BalitaLansia_Desa', 0)):,}</b><br>"
                f"&nbsp;&nbsp;• Warga Miskin: <b>{int(d.get('Rentan_Miskin_Desa', 0)):,}</b><br>"
                f"&nbsp;&nbsp;• Disabilitas: <b>{int(d.get('Rentan_Disabilitas_Desa', 0)):,}</b><br>"
                f"&nbsp;&nbsp;<i>Total: {_desa_total_rentan:,} jiwa</i><br>"
                f"🎯 Prioritas Warga Teredukasi: "
                f"<b style=\"color:#d35400;\">{persen_teredukasi:.2f}%</b><br>"
                # Keterangan kecil (permintaan BPBD): jelaskan basis
                # hitung persentase. Level desa = porsi satu desa
                # terhadap total kecamatan induknya.
                f"<span style=\"font-size:10px;color:#7f8c8d;\">"
                f"<i>Persentase satu desa dari total Kec. "
                f"{d.get('Kecamatan', '')}</i></span>"
                + _tt_wrap_close_d
            )

            tooltip_map_desa[key_d]['destana'] = (
                _tt_wrap_open_d + _tt_head_desa +
                f"👥 Warga Terpapar: <b>{_desa_terpapar:,}</b> jiwa<br>"
                f"🛡️ Skor Destana: <b style=\"color:{warna_destana_desa};\">"
                f"{skor_destana_desa:.2f}%</b> "
                f"<span style=\"font-size:10px;color:{warna_destana_desa};\">"
                f"({label_destana_desa})</span>"
                f"<div style=\"font-size:10px;margin-top:4px;\">"
                f"<u>Sudah dicentang:</u>{_list_dicentang_html}</div>"
                f"<div style=\"font-size:10px;margin-top:4px;\">"
                f"<u>Kurang:</u>{_list_kurang_html}</div>"
                + _tt_wrap_close_d
            )

            tooltip_html = tooltip_map_desa[key_d]['realisasi']
 
 
            desa_kecamatan = clean_name(
                d.get(
                    "Kecamatan",
                    ""
                )
            )
 
 
        else:
 
            popup_html = (
                f"<b>Desa: "
                f"{raw_desa}</b>"
            )
 
            tooltip_html = (
                f"Desa: "
                f"{raw_desa}"
            )
 
            desa_kecamatan = clean_name(
                get_nama_kecamatan(
                    feature["properties"]
                )
            )
 
 
        folium.GeoJson(
 
            feature,
 
            style_function=
                style_desa,
 
            tooltip=folium.Tooltip(
                tooltip_html
            ),
 
            popup=folium.Popup(
                popup_html,
                max_width=350
            ),
 
            name=
                f"desa_{desa_kecamatan}"
 
        ).add_to(m)
 
 
    # ====================================================
    # 3. LEGEND
    # ====================================================
 
    overlay_html = """

    <!-- =============================================
         PANEL KIRI-BAWAH: tombol reset + legend
         (dibungkus 1 container flex-column supaya
         tombol reset otomatis nempel PERSIS di atas
         legend yang lagi aktif, dan gak numpuk sama
         panel chatbot yang ada di kanan-atas)
    ============================================== -->

    <div
        id="panel-kiri-bawah"
        style="
            position:absolute;
            bottom:40px;
            left:20px;
            z-index:99999;
            display:flex;
            flex-direction:column;
            align-items:stretch;
            gap:10px;
        "
    >

    <button
        id="btn-reset-map"
        onclick="resetToKecamatanView()"
        style="
            display:none;
            background:#0f172a;
            color:#00ffcc;
            border:1px solid #00ffcc;
            padding:10px 15px;
            border-radius:6px;
            cursor:pointer;
            font-weight:bold;
        "
    >

        ↺ KEMBALI KE OVERVIEW KECAMATAN

    </button>


    <!-- =============================================
         DROPDOWN JENIS BENCANA (khusus tab Kajian Risiko Bencana)
         Muncul di atas tombol tab hanya saat tab itu aktif; isi
         opsi dari kajian_risiko.opsi_dropdown_html().
    ============================================== -->

    <div
        id="kajian-picker"
        style="
            display:none;
            align-items:center;
            gap:8px;
            max-width:520px;
            background-color:rgba(14,17,23,0.92);
            border:1px solid #00ffcc;
            border-radius:8px;
            padding:6px 10px;
            font-family:monospace;
            font-size:11px;
            color:#00ffcc;
            box-shadow:0 0 15px rgba(0,255,204,0.12);
        "
    >
        <label for="select-bencana-kajian" style="font-weight:bold;white-space:nowrap;">Jenis Bencana:</label>
        <select
            id="select-bencana-kajian"
            onchange="setKajianHazard(this.value)"
            style="
                flex:1;
                min-width:0;
                background:#0f172a;
                color:#fff;
                border:1px solid #00ffcc;
                border-radius:4px;
                padding:4px 6px;
                font-family:monospace;
                font-size:11px;
                cursor:pointer;
            "
        >__KAJIAN_OPSI_HTML__</select>
    </div>


    <!-- =============================================
         TOGGLE MODE ANALISIS PETA
         (realisasi, prioritas, atau skor Destana)
    ============================================== -->

    <div
        id="toggle-mode-warna"
        style="
            display:flex;
            max-width:520px;
            border:1px solid #00ffcc;
            border-radius:8px;
            overflow:hidden;
            font-family:monospace;
            font-size:11px;
            box-shadow:0 0 15px rgba(0,255,204,0.12);
        "
    >
        <button
            id="btn-mode-realisasi"
            onclick="applyColorMode('realisasi')"
            style="
                flex:1;
                padding:8px 6px;
                border:none;
                cursor:pointer;
                font-weight:bold;
                background:#00ffcc;
                color:#0f172a;
            "
        >📊 Realisasi Edukasi</button>
        <button
            id="btn-mode-prioritas"
            onclick="applyColorMode('prioritas')"
            style="
                flex:1;
                padding:8px 6px;
                border:none;
                cursor:pointer;
                font-weight:bold;
                background:#0f172a;
                color:#00ffcc;
            "
        >🎯 Prioritas Edukasi</button>
        <button id="btn-mode-destana" onclick="applyColorMode('destana')" title="Warna menunjukkan rata-rata skor Destana desa dalam kecamatan" style="flex:1;padding:8px 6px;border:none;cursor:pointer;font-weight:bold;background:#0f172a;color:#00ffcc;">🛡️ Destana</button>
        <button id="btn-mode-kajian-risiko" onclick="applyColorMode('kajian_risiko')" title="Tingkat risiko per jenis bencana, langsung dari tabulasi Kajian Risiko Bencana BPBD. Pilih jenis bencana lewat dropdown yang muncul di atas tombol ini." style="flex:1.3;padding:8px 6px;border:none;cursor:pointer;font-weight:bold;background:#0f172a;color:#00ffcc;">🗺️ Kajian Risiko Bencana</button>
    </div>


    <!-- =============================================
         LEGEND KECAMATAN
    ============================================== -->

    <div
        id="legend-kecamatan"
        style="
            width:230px;
            background-color:rgba(14,17,23,0.9);
            color:#fff;
            font-size:12px;
            font-family:monospace;
            border:1px solid #00ffcc;
            border-radius:8px;
            padding:12px;
            box-shadow:0 0 15px rgba(0,255,204,0.2);
            backdrop-filter:blur(5px);
        "
    >

        <b id="legend-kec-title" style="
            font-size:13px;
            color:#00ffcc;
        ">
            Rasio Realisasi Edukasi
            (Kecamatan)
        </b>

        <br>

        <span id="legend-kec-subtitle" style="
            font-size:10px;
            color:#cbd5e1;
        ">
            Warga Teredukasi / Warga
            Prioritas (Wajib Edukasi) x 100%
        </span>

        <hr style="
            margin:6px 0;
            border:0;
            border-top:
                1px solid
                rgba(0,255,204,0.3);
        ">

        <div id="legend-kec-bar-wrap" style="
            position:relative;
            height:14px;
            border-radius:4px;
            margin-bottom:6px;
        ">
            <div style="
                height:100%;
                border-radius:4px;
                background:linear-gradient(
                    to right,
                    #d7191c 0%,
                    #ffffbf 50%,
                    #1a9641 100%
                );
            "></div>
            <div id="legend-kec-threshold" style="
                display:none;
                position:absolute;
                top:-3px;
                bottom:-3px;
                width:2px;
                background:#ffffff;
                box-shadow:0 0 4px rgba(255,255,255,0.9);
            "></div>
        </div>

        <div id="legend-kec-threshold-caption" style="
            display:none;
            font-size:9px;
            color:#e2e8f0;
            text-align:center;
            margin:-2px 0 6px 0;
        "></div>

        <div id="legend-kec-minmax" style="
            display:flex;
            justify-content:space-between;
            font-size:10px;
            color:#cbd5e1;
        ">
            <span id="legend-kec-min">0% <br><i>(belum ada yang teredukasi)</i></span>
            <span id="legend-kec-max" style="text-align:right;">100% <br><i>(semua warga prioritas teredukasi)</i></span>
        </div>

        <!-- Legenda kategori (Rendah/Sedang/Tinggi) khusus tab Kajian Risiko Bencana;
             isinya diisi JS (kajianAturLegenda) -->
        <div id="legend-kec-kajian" style="display:none;"></div>

        <div id="legend-kec-status-note" style="
            display:none;
            margin-top:8px;
            padding-top:6px;
            border-top:1px dashed rgba(0,255,204,0.25);
            font-size:10px;
            color:#a7f3d0;
        "></div>

    </div>


    <!-- =============================================
         LEGEND DESA
    ============================================== -->

    <div
        id="legend-desa"
        style="
            display:none;
            width:250px;
            background-color:rgba(14,17,23,0.9);
            color:#fff;
            font-size:12px;
            font-family:monospace;
            border:1px solid #00ffcc;
            border-radius:8px;
            padding:12px;
            box-shadow:0 0 15px rgba(0,255,204,0.2);
            backdrop-filter:blur(5px);
        "
    >

        <b id="legend-desa-title" style="
            font-size:13px;
            color:#00ffcc;
        ">
            Rasio Realisasi Edukasi
            (Desa)
        </b>

        <br>

        <span id="legend-desa-subtitle" style="
            font-size:10px;
            color:#cbd5e1;
        ">
            Warga Teredukasi / Warga
            Prioritas (Wajib Edukasi) x 100%
        </span>

        <hr style="
            margin:6px 0;
            border:0;
            border-top:
                1px solid
                rgba(0,255,204,0.3);
        ">

        <div id="legend-desa-bar-wrap" style="
            position:relative;
            height:14px;
            border-radius:4px;
            margin-bottom:6px;
        ">
            <div style="
                height:100%;
                border-radius:4px;
                background:linear-gradient(
                    to right,
                    #d7191c 0%,
                    #ffffbf 50%,
                    #1a9641 100%
                );
            "></div>
            <div id="legend-desa-threshold" style="
                display:none;
                position:absolute;
                top:-3px;
                bottom:-3px;
                width:2px;
                background:#ffffff;
                box-shadow:0 0 4px rgba(255,255,255,0.9);
            "></div>
        </div>

        <div id="legend-desa-threshold-caption" style="
            display:none;
            font-size:9px;
            color:#e2e8f0;
            text-align:center;
            margin:-2px 0 6px 0;
        "></div>

        <div id="legend-desa-minmax" style="
            display:flex;
            justify-content:space-between;
            font-size:10px;
            color:#cbd5e1;
        ">
            <span id="legend-desa-min">0% <br><i>(belum ada yang teredukasi)</i></span>
            <span id="legend-desa-max" style="text-align:right;">100% <br><i>(semua warga prioritas teredukasi)</i></span>
        </div>

        <!-- Legenda kategori (Rendah/Sedang/Tinggi) khusus tab Kajian Risiko Bencana;
             isinya diisi JS (kajianAturLegenda) -->
        <div id="legend-desa-kajian" style="display:none;"></div>

        <div id="legend-desa-status-note" style="
            display:none;
            margin-top:8px;
            padding-top:6px;
            border-top:1px dashed rgba(0,255,204,0.25);
            font-size:10px;
            color:#a7f3d0;
        "></div>

    </div>

    </div>
    <!-- /#panel-kiri-bawah -->

    """


    # Opsi dropdown "Jenis Bencana" (tab Kajian Risiko Bencana)
    overlay_html = overlay_html.replace(
        '__KAJIAN_OPSI_HTML__',
        kajian_risiko.opsi_dropdown_html()
    )

    m.get_root().html.add_child(
        folium.Element(
            overlay_html
        )
    )
 
 
    # ====================================================
    # 4. JAVASCRIPT INTERAKSI MAP
    # ====================================================
 
    interactive_script = """
 
    <style>
 
        .leaflet-container
        path.layer-desa {
 
            display:none !important;
 
        }
 
 
        .label-teks-daerah {
 
            background:none;
            border:none;
 
            color:#0f172a;
 
            font-size:11px;
 
            font-weight:800;
 
            text-align:center;
 
            text-shadow:
                1px 1px 2px #fff,
                -1px -1px 2px #fff,
                1px -1px 2px #fff,
                -1px 1px 2px #fff;
 
            pointer-events:none;
 
            white-space:nowrap;
 
        }
 
    </style>
 
 
    <script>
 
        let mainMapObj = null;


        // ------------------------------------------------
        // PETA WARNA 3 MODE (dikirim dari server sebagai JSON,
        // key-nya = clean_name desa/kecamatan, sama persis dengan
        // yang dipakai backend). Dipakai oleh applyColorMode() di
        // bawah untuk toggle warna TANPA reload halaman.
        // ------------------------------------------------
        let colorMapDesa = __COLOR_MAP_DESA_JSON__;
        let colorMapKec = __COLOR_MAP_KEC_JSON__;
        let colorMapDestanaDesa = __COLOR_MAP_DESTANA_DESA_JSON__;
        let colorMapDestanaKec = __COLOR_MAP_DESTANA_KEC_JSON__;
        let destanaScoreDesa = __DESTANA_SCORE_DESA_JSON__;
        let destanaScoreKec = __DESTANA_SCORE_KEC_JSON__;
        let destanaStatusKec = __DESTANA_STATUS_KEC_JSON__;
        let destanaThreshold = __DESTANA_THRESHOLD__;

        // Tooltip per mode/tab (realisasi, prioritas, destana;
        // tab kajian_risiko dibangun terpisah oleh kajian_risiko_map.js),
        // key-nya sama dengan colorMapDesa/Kec di atas. Dipakai applyColorMode() supaya isi tooltip ikut
        // berganti sesuai tab yang aktif -- bukan cuma warnanya saja.
        let tooltipMapDesa = __TOOLTIP_MAP_DESA_JSON__;
        let tooltipMapKec = __TOOLTIP_MAP_KEC_JSON__;

        let currentColorMode = 'realisasi';

        __KAJIAN_RISIKO_JS__
 
 
        let initialCenter = [
            -6.8452,
            107.5023
        ];
 
 
        let initialZoom = 11;
 
 
        let labelKecamatanLayer =
            L.layerGroup();
 
 
        let labelDesaLayer =
            L.layerGroup();
 
 
        document.addEventListener(
            "DOMContentLoaded",
            function() {
 
                let checkMapInterval =
                    setInterval(
                        function() {
 
                            for (
                                let key in window
                            ) {
 
                                if (
                                    key.startsWith(
                                        "map_"
                                    )
                                    &&
                                    window[key]
                                    &&
                                    window[key].on
                                ) {
 
                                    clearInterval(
                                        checkMapInterval
                                    );
 
                                    mainMapObj =
                                        window[key];
 
                                    initInteractiveMap();

                                    applyColorMode(currentColorMode);
 
                                    break;
                                }
                            }
 
                        },
                        300
                    );
 
            }
        );
 
 
        function cleanNameJS(name) {
 
            if (!name)
                return '';
 
 
            return String(name)
                .toLowerCase()
                .replace(
                    /kecamatan/g,
                    ''
                )
                .replace(
                    /kec\\.?/g,
                    ''
                )
                .replace(
                    /desa/g,
                    ''
                )
                .replace(
                    /[^a-z0-9]/g,
                    ''
                );
 
        }
 
 
        // ------------------------------------------------
        // TOGGLE MODE WARNA PETA (realisasi <-> prioritas)
        // ------------------------------------------------
        // Warna tiap desa/kecamatan sudah dihitung SEKALI di server
        // (colorMapDesa / colorMapKec, dikirim sebagai JSON), jadi
        // ganti mode di sini murni operasi visual client-side lewat
        // layer.setStyle() -- instan, tanpa panggil server lagi.
        // ------------------------------------------------

        function applyColorMode(mode) {

            currentColorMode = mode;

            if (!mainMapObj) return;

            mainMapObj.eachLayer(function(layer) {

                // ------------------------------------------------
                // PENTING: Folium mengikat tooltip ke OBJEK
                // PEMBUNGKUS (wrapper) folium.GeoJson(...), bukan ke
                // layer path/polygon anaknya. Tapi layer path/polygon
                // anaknya-lah yang punya properti ".feature" (dipakai
                // buat setStyle warna & buat cari key desa/kecamatan).
                //
                // Makanya di sini kita pisahkan:
                //   - featureLayer -> sumber .feature/properties buat
                //     tahu ini desa/kec mana (dipakai buat key & warna)
                //   - layer (parameter asli) -> objek yang beneran
                //     dipakai buat setTooltipContent(), KALAU dia yang
                //     punya tooltip terpasang (dicek via getTooltip())
                //
                // Kalau cuma cek "layer.feature" seperti sebelumnya,
                // wrapper (yang justru punya tooltip) malah ke-skip,
                // sehingga setTooltipContent() dipanggil ke layer anak
                // yang TIDAK punya tooltip -> tidak berefek apa-apa
                // (makanya isi tooltip kelihatan "sama saja" di semua
                // tab kemarin).
                // ------------------------------------------------

                let featureLayer = null;

                if (layer.feature && layer.feature.properties) {
                    featureLayer = layer;
                } else if (layer.getLayers) {
                    let children = layer.getLayers();
                    for (let i = 0; i < children.length; i++) {
                        if (children[i].feature && children[i].feature.properties) {
                            featureLayer = children[i];
                            break;
                        }
                    }
                }

                if (!featureLayer) return;

                let properties = featureLayer.feature.properties;

                let className =
                    featureLayer.options && featureLayer.options.className
                        ? featureLayer.options.className
                        : "";

                // Objek yang benar-benar punya tooltip terpasang bisa
                // "layer" (wrapper) ATAU "featureLayer" (anak),
                // tergantung versi folium -- jadi cek keduanya supaya
                // aman di kedua kondisi.
                // Tooltip awal (sebelum ditimpa mode kajian) disimpan supaya desa
                // yang TIDAK punya entri tooltip di mode lain (mis. desa yang
                // tidak ada di data edukasi) bisa dikembalikan.
                // Tooltip bisa menempel di pembungkus (layer) ATAU di
                // poligon anaknya (featureLayer), jadi disimpan/dipulihkan
                // per objek yang benar-benar memegang tooltip.
                function simpanTooltipAwal() {
                    [layer, featureLayer].forEach(function (o) {
                        if (o._tipAsli === undefined && o.getTooltip && o.getTooltip()) {
                            o._tipAsli = o.getTooltip().getContent();
                        }
                    });
                }

                function pulihkanTooltipAwal() {
                    [layer, featureLayer].forEach(function (o) {
                        if (o._tipAsli !== undefined && o.getTooltip && o.getTooltip() && o.setTooltipContent) {
                            o.setTooltipContent(o._tipAsli);
                        }
                    });
                }

                function updateTooltipTo(newHtml) {
                    if (layer.getTooltip && layer.getTooltip() && layer.setTooltipContent) {
                        layer.setTooltipContent(newHtml);
                    }
                    if (
                        featureLayer !== layer &&
                        featureLayer.getTooltip && featureLayer.getTooltip() &&
                        featureLayer.setTooltipContent
                    ) {
                        featureLayer.setTooltipContent(newHtml);
                    }
                }

                if (className.includes("layer-kecamatan")) {

                    let namaKec =
                        properties.nm_kecamatan ||
                        properties.Kecamatan ||
                        properties.kecamatan;

                    if (!namaKec) return;

                    let key = cleanNameJS(namaKec);

                    if (featureLayer.setStyle) {
                        let warna = null;
                        if (mode === 'destana') {
                            warna = colorMapDestanaKec[key] || null;
                        } else if (mode === 'kajian_risiko') {
                            if (featureLayer._fillAsli === undefined) featureLayer._fillAsli = featureLayer.options.fillColor;
                            warna = kajianFillColor('kec', key);
                        } else if (colorMapKec[key]) {
                            warna = colorMapKec[key][mode];
                        }
                        if (warna) featureLayer.setStyle({ fillColor: warna });
                        else if (featureLayer._fillAsli !== undefined) featureLayer.setStyle({ fillColor: featureLayer._fillAsli });
                    }

                    if (mode === 'kajian_risiko') {
                        simpanTooltipAwal();
                        updateTooltipTo(kajianTooltipHtml('kec', key));
                    } else if (tooltipMapKec[key] && tooltipMapKec[key][mode]) {
                        updateTooltipTo(tooltipMapKec[key][mode]);
                    } else {
                        pulihkanTooltipAwal();
                    }

                }

                if (className.includes("layer-desa")) {

                    let namaDesa =
                        properties.village ||
                        properties.nama_desa ||
                        properties.desa ||
                        properties.nm_desa ||
                        properties.DESA ||
                        properties.NAMOBJ;

                    if (!namaDesa) return;

                    let key = cleanNameJS(namaDesa);

                    if (featureLayer.setStyle) {
                        // Warna awal disimpan sebelum tab Kajian menimpanya, supaya
                        // desa yang TIDAK punya warna di tab lain (mis. tidak ada
                        // di data edukasi, abu-abu #cccccc) bisa dikembalikan.
                        let warna = null;
                        if (mode === 'destana') {
                            warna = colorMapDestanaDesa[key] || null;
                        } else if (mode === 'kajian_risiko') {
                            if (featureLayer._fillAsli === undefined) featureLayer._fillAsli = featureLayer.options.fillColor;
                            warna = kajianFillColor('desa', kajianKeyDesa(properties));
                        } else if (colorMapDesa[key]) {
                            warna = colorMapDesa[key][mode];
                        }
                        if (warna) featureLayer.setStyle({ fillColor: warna });
                        else if (featureLayer._fillAsli !== undefined) featureLayer.setStyle({ fillColor: featureLayer._fillAsli });
                    }

                    if (mode === 'kajian_risiko') {
                        simpanTooltipAwal();
                        updateTooltipTo(kajianTooltipHtml('desa', kajianKeyDesa(properties)));
                    } else if (tooltipMapDesa[key] && tooltipMapDesa[key][mode]) {
                        updateTooltipTo(tooltipMapDesa[key][mode]);
                    } else {
                        pulihkanTooltipAwal();
                    }

                }

            });

            updateLegendText(mode);
            updateToggleButtons(mode);

        }


        function updateLegendText(mode) {

            let judul, subjudul, labelMin, labelMax;

            if (mode === 'destana') {
                judul = 'Skor Destana';
                subjudul = 'Rata-rata persentase indikator Destana yang terpenuhi';
                labelMin = '0% <br><i>(kesiapsiagaan rendah)</i>';
                labelMax = '100% <br><i>(seluruh indikator terpenuhi)</i>';
            } else if (mode === 'kajian_risiko') {
                // Judul & isi legenda kategori diatur kajian_risiko_map.js
                let kj = kajianJudulLegenda();
                judul = kj.judul;
                subjudul = kj.subjudul;
                labelMin = '';
                labelMax = '';
            } else if (mode === 'prioritas') {
                judul = 'Prioritas Edukasi Tersisa';
                subjudul = 'Jumlah Warga Prioritas yang Belum Diedukasi (Wajib Edukasi - Sudah Diedukasi)';
                labelMin = 'Sedikit <br><i>(sisa prioritas paling kecil)</i>';
                labelMax = 'Banyak <br><i>(sisa prioritas paling besar)</i>';
            } else {
                judul = 'Rasio Realisasi Edukasi';
                subjudul = 'Warga Teredukasi / Warga Prioritas (Wajib Edukasi) x 100%';
                labelMin = '0% <br><i>(belum ada yang teredukasi)</i>';
                labelMax = '100% <br><i>(semua warga prioritas teredukasi)</i>';
            }

            let idPairs = [
                ['legend-kec-title', judul + ' (Kecamatan)'],
                ['legend-kec-subtitle', subjudul],
                ['legend-desa-title', judul + ' (Desa)'],
                ['legend-desa-subtitle', subjudul],
            ];

            idPairs.forEach(function (pair) {
                let el = document.getElementById(pair[0]);
                if (el) el.innerHTML = pair[1];
            });

            let htmlPairs = [
                ['legend-kec-min', labelMin],
                ['legend-kec-max', labelMax],
                ['legend-desa-min', labelMin],
                ['legend-desa-max', labelMax],
            ];

            htmlPairs.forEach(function (pair) {
                let el = document.getElementById(pair[0]);
                if (el) el.innerHTML = pair[1];
            });

            // ------------------------------------------------
            // Penanda batas status "Destana" (garis putih di
            // atas gradasi + catatan jumlah kecamatan/desa yang
            // sudah lolos threshold). Cuma relevan buat mode
            // 'destana' -- mode realisasi & prioritas gak punya
            // konsep "lulus/belum lulus", jadi disembunyikan lagi
            // begitu mode lain dipilih.
            // ------------------------------------------------

            let kecBar        = document.getElementById('legend-kec-threshold');
            let kecCaption    = document.getElementById('legend-kec-threshold-caption');
            let kecStatusNote = document.getElementById('legend-kec-status-note');
            let desaBar        = document.getElementById('legend-desa-threshold');
            let desaCaption    = document.getElementById('legend-desa-threshold-caption');
            let desaStatusNote = document.getElementById('legend-desa-status-note');

            if (mode === 'destana') {

                let posisi = Math.max(0, Math.min(100, destanaThreshold));

                if (kecBar)  { kecBar.style.display  = 'block'; kecBar.style.left  = posisi + '%'; }
                if (desaBar) { desaBar.style.display = 'block'; desaBar.style.left = posisi + '%'; }

                let capTxt = '▲ Batas status &quot;Destana&quot;: ' + destanaThreshold + '%';
                if (kecCaption)  { kecCaption.style.display  = 'block'; kecCaption.innerHTML  = capTxt; }
                if (desaCaption) { desaCaption.style.display = 'block'; desaCaption.innerHTML = capTxt; }

                let kecValues  = Object.values(destanaScoreKec || {});
                let desaValues = Object.values(destanaScoreDesa || {});

                let kecLulus  = kecValues.filter(function (v) { return v >= destanaThreshold; }).length;
                let desaLulus = desaValues.filter(function (v) { return v >= destanaThreshold; }).length;

                if (kecStatusNote) {
                    kecStatusNote.style.display = 'block';
                    kecStatusNote.innerHTML =
                        '🛡️ ' + kecLulus + ' dari ' + kecValues.length +
                        ' kecamatan sudah berstatus <b>Destana</b>';
                }

                if (desaStatusNote) {
                    desaStatusNote.style.display = 'block';
                    desaStatusNote.innerHTML =
                        '🛡️ ' + desaLulus + ' dari ' + desaValues.length +
                        ' desa sudah berstatus <b>Destana</b>';
                }

            } else {

                if (kecBar)  kecBar.style.display  = 'none';
                if (desaBar) desaBar.style.display = 'none';
                if (kecCaption)  kecCaption.style.display  = 'none';
                if (desaCaption) desaCaption.style.display = 'none';
                if (kecStatusNote)  kecStatusNote.style.display  = 'none';
                if (desaStatusNote) desaStatusNote.style.display = 'none';

            }

            // Legenda kategori Rendah/Sedang/Tinggi (khusus tab Kajian
            // Risiko) menggantikan gradasi + label min/max bawaan.
            kajianAturLegenda(mode);

        }


        function updateToggleButtons(mode) {

            let buttons = {
                realisasi: document.getElementById('btn-mode-realisasi'),
                prioritas: document.getElementById('btn-mode-prioritas'),
                destana: document.getElementById('btn-mode-destana'),
                kajian_risiko: document.getElementById('btn-mode-kajian-risiko')
            };
            kajianAturDropdown(mode);
            if (!buttons.realisasi || !buttons.prioritas || !buttons.destana || !buttons.kajian_risiko) return;
            Object.keys(buttons).forEach(function(key) {
                buttons[key].style.background = key === mode ? '#00ffcc' : '#0f172a';
                buttons[key].style.color = key === mode ? '#0f172a' : '#00ffcc';
            });

        }


        function initInteractiveMap() {

 
            mainMapObj.addLayer(
                labelKecamatanLayer
            );
 
 
            mainMapObj.eachLayer(
                function(layer) {
 
                    if (
                        layer.feature
                        &&
                        layer.feature.properties
                        &&
                        layer.getBounds
                    ) {
 
                        let properties =
                            layer.feature.properties;
 
 
                        let className =
                            layer.options
                            &&
                            layer.options.className
                            ?
                            layer.options.className
                            :
                            "";
 
 
                        // ==================================
                        // KECAMATAN
                        // ==================================
 
                        if (
                            className.includes(
                                "layer-kecamatan"
                            )
                        ) {
 
                            let namaKecamatan =
                                properties.nm_kecamatan;
 
 
                            if (!namaKecamatan) {
 
                                namaKecamatan =
                                    properties.Kecamatan
                                    ||
                                    properties.kecamatan;
 
                            }
 
 
                            if (namaKecamatan) {
 
                                let center =
                                    layer
                                        .getBounds()
                                        .getCenter();
 
 
                                let textMarker =
                                    L.marker(
                                        center,
                                        {
                                            icon:
                                                L.divIcon(
                                                    {
                                                        className:
                                                            "label-teks-daerah",
 
                                                        html:
                                                            namaKecamatan
                                                                .toUpperCase()
                                                    }
                                                ),
 
                                            interactive:
                                                false
                                        }
                                    );
 
 
                                labelKecamatanLayer
                                    .addLayer(
                                        textMarker
                                    );
 
 
                                layer.on(
                                    "click",
                                    function(e) {
 
                                        let cleanKecName =
                                            cleanNameJS(
                                                namaKecamatan
                                            );
 
 
                                        showDesaForKecamatan(
                                            cleanKecName,
                                            layer.getBounds()
                                        );
 
                                    }
                                );
 
                            }
 
                        }
 
 
                        // ==================================
                        // DESA
                        // ==================================
 
                        if (
                            className.includes(
                                "layer-desa"
                            )
                        ) {
 
                            let namaDesa =
                                properties.village
                                ||
                                properties.nama_desa
                                ||
                                properties.desa
                                ||
                                properties.nm_desa
                                ||
                                properties.DESA
                                ||
                                properties.NAMOBJ;
 
 
                            let namaKecParent =
                                properties.nm_kecamatan
                                ||
                                properties.Kecamatan
                                ||
                                properties.kecamatan;
 
 
                            if (namaDesa) {
 
                                let center =
                                    layer
                                        .getBounds()
                                        .getCenter();
 
 
                                let textMarker =
                                    L.marker(
                                        center,
                                        {
                                            icon:
                                                L.divIcon(
                                                    {
                                                        className:
                                                            "label-teks-daerah",
 
                                                        html:
                                                            namaDesa
                                                                .toUpperCase()
                                                    }
                                                ),
 
                                            interactive:
                                                false
                                        }
                                    );
 
 
                                textMarker.kecParentId =
                                    cleanNameJS(
                                        namaKecParent
                                    );
 
 
                                labelDesaLayer
                                    .addLayer(
                                        textMarker
                                    );
 
                            }
 
                        }
 
                    }
 
                }
            );
 
        }
 
 
        function showDesaForKecamatan(
            cleanKecName,
            bounds
        ) {

            // Legenda kajian risiko menghitung desa di kecamatan yang
            // sedang dibuka saja.
            kajianKecTerbuka = cleanKecName;
            if (currentColorMode === 'kajian_risiko') {
                kajianAturLegenda(currentColorMode);
            }
 
            document
                .querySelectorAll(
                    ".layer-kecamatan"
                )
                .forEach(
                    function(el) {
 
                        el.style.setProperty(
                            "display",
                            "none",
                            "important"
                        );
 
                    }
                );
 
 
            mainMapObj.removeLayer(
                labelKecamatanLayer
            );
 
 
            mainMapObj.addLayer(
                labelDesaLayer
            );
 
 
            labelDesaLayer.eachLayer(
                function(marker) {
 
                    if (
                        marker.kecParentId ===
                        cleanKecName
                    ) {
 
                        marker.setOpacity(1);
 
                    }
                    else {
 
                        marker.setOpacity(0);
 
                    }
 
                }
            );
 
 
            mainMapObj.eachLayer(
                function(layer) {
 
                    if (
                        layer.feature
                        &&
                        layer.feature.properties
                    ) {
 
                        let properties =
                            layer.feature.properties;
 
 
                        let namaKec =
                            properties.nm_kecamatan
                            ||
                            properties.Kecamatan
                            ||
                            properties.kecamatan;
 
 
                        if (!namaKec)
                            return;
 
 
                        if (
                            cleanNameJS(
                                namaKec
                            )
                            ===
                            cleanKecName
                        ) {
 
                            if (
                                layer.getElement()
                            ) {
 
                                layer.getElement()
                                    .style
                                    .setProperty(
                                        "display",
                                        "block",
                                        "important"
                                    );
 
                            }
 
                        }
 
                    }
 
                }
            );
 
 
            document.getElementById(
                "legend-kecamatan"
            ).style.display = "none";
 
 
            document.getElementById(
                "legend-desa"
            ).style.display = "block";
 
 
            document.getElementById(
                "btn-reset-map"
            ).style.display = "block";
 
 
            if (
                mainMapObj
                &&
                bounds
            ) {
 
                mainMapObj.fitBounds(
                    bounds,
                    {
                        padding: [
                            20,
                            20
                        ]
                    }
                );
 
            }
 
        }
 
 
        function resetToKecamatanView() {

            kajianKecTerbuka = null;
            if (currentColorMode === 'kajian_risiko') {
                kajianAturLegenda(currentColorMode);
            }
 
            document
                .querySelectorAll(
                    ".layer-kecamatan"
                )
                .forEach(
                    function(el) {
 
                        el.style.setProperty(
                            "display",
                            "block",
                            "important"
                        );
 
                    }
                );
 
 
            document
                .querySelectorAll(
                    ".layer-desa"
                )
                .forEach(
                    function(el) {
 
                        el.style.setProperty(
                            "display",
                            "none",
                            "important"
                        );
 
                    }
                );
 
 
            mainMapObj.removeLayer(
                labelDesaLayer
            );
 
 
            mainMapObj.addLayer(
                labelKecamatanLayer
            );
 
 
            document.getElementById(
                "legend-kecamatan"
            ).style.display = "block";
 
 
            document.getElementById(
                "legend-desa"
            ).style.display = "none";
 
 
            document.getElementById(
                "btn-reset-map"
            ).style.display = "none";
 
 
            if (mainMapObj) {
 
                mainMapObj.setView(
                    initialCenter,
                    initialZoom
                );
 
            }
 
        }
 
    </script>
 
    """

    # Sisipkan blok JS tab Kajian Risiko Bencana, lalu isi placeholder
    # datanya (JSON dari tabulasi -- lihat kajian_risiko.data_js()).
    interactive_script = interactive_script.replace(
        '__KAJIAN_RISIKO_JS__', _js_kajian_risiko()
    )
    for _ph, _val in kajian_risiko.data_js(kajian_data_peta).items():
        interactive_script = interactive_script.replace(_ph, _val)

    interactive_script = (
        interactive_script
        .replace('__COLOR_MAP_DESA_JSON__', json.dumps(color_map_desa))
        .replace('__COLOR_MAP_KEC_JSON__', json.dumps(color_map_kec))
        .replace('__COLOR_MAP_DESTANA_DESA_JSON__', json.dumps(color_map_destana_desa))
        .replace('__COLOR_MAP_DESTANA_KEC_JSON__', json.dumps(color_map_destana_kec))
        .replace('__DESTANA_SCORE_DESA_JSON__', json.dumps(destana_score_desa))
        .replace('__DESTANA_SCORE_KEC_JSON__', json.dumps(destana_score_kec))
        .replace('__DESTANA_STATUS_KEC_JSON__', json.dumps(destana_status_kec))
        .replace('__DESTANA_THRESHOLD__', str(threshold_now))
        .replace('__TOOLTIP_MAP_DESA_JSON__', json.dumps(tooltip_map_desa))
        .replace('__TOOLTIP_MAP_KEC_JSON__', json.dumps(tooltip_map_kec))
    )
 
 
    m.get_root().html.add_child(
        folium.Element(
            interactive_script
        )
    )
 
 
    # ====================================================
    # RENDER PETA KE IFRAME
    # ====================================================
    #
    # CATATAN PENTING (perbaikan bug "kepotong/crop"):
    #
    # Sebelumnya di sini dipakai m._repr_html_(), yaitu method
    # bawaan Folium yang didesain untuk tampil di Jupyter Notebook.
    # Method itu membungkus peta dengan trik CSS
    # "padding-bottom: 60%" supaya tingginya proporsional
    # terhadap LEBAR peta (60% dari lebar) -- BUKAN mengikuti
    # tinggi kontainer aslinya.
    #
    # Di kondisi.html, .map-container memakai flex-grow untuk
    # mengisi SISA tinggi layar (bisa jauh lebih besar/kecil dari
    # 60% lebarnya, tergantung ukuran layar). Kalau tinggi asli
    # kontainer lebih kecil dari hasil trik 60% itu, maka bagian
    # bawah peta (termasuk tombol reset & legend yang nempel di
    # posisi bottom/top absolute di dalam iframe) jadi terpotong,
    # karena body halaman pakai overflow:hidden (tidak bisa di-scroll).
    #
    # Perbaikannya: render HTML peta secara manual lalu bungkus
    # sendiri dalam iframe dengan width:100% & height:100% supaya
    # iframe benar-benar mengikuti ukuran .map-container yang
    # sesungguhnya, bukan rasio tebakan dari lebar.
    # ====================================================

    map_source = m.get_root().render()
    map_source_escaped = html.escape(map_source, quote=True)

    return (
        '<iframe srcdoc="' + map_source_escaped + '" '
        'style="width:100%;height:100%;border:none;display:block;" '
        'allowfullscreen webkitallowfullscreen mozallowfullscreen>'
        '</iframe>'
    )
 
 

# ========================================================
# DESTANA
# ========================================================
# Kategori berikut berasal dari DATA DESTANA 2026 yang diberikan.
# PDF hanya memuat kategori (Pratama/Madya), bukan definisi indikator.
DESTANA_KATEGORI_2026 = {
    "cikahuripan": "Madya", "langensari": "Pratama", "sukajaya": "Pratama",
    "kayuambon": "Pratama", "wangunsari": "Pratama", "mekarwangi": "Pratama",
    "wangunharja": "Pratama", "cibogo": "Pratama", "pagerwangi": "Pratama",
    "gudangkahuripan": "Pratama", "suntenjaya": "Pratama", "cikole": "Pratama",
    "cibodas": "Pratama", "lembang": "Pratama", "cikidang": "Pratama",
    "jayagiri": "Pratama", "tugumukti": "Pratama", "pasirlangu": "Pratama",
    "cipada": "Pratama", "jambudipa": "Pratama", "kertawangi": "Pratama",
    "padaasih": "Pratama", "pasirhalang": "Pratama", "sadangmekar": "Pratama",
    "batulayang": "Pratama", "mukapayung": "Pratama",
    "cicangkanggirang": "Pratama", "bojongkoneng": "Pratama", "cilame": "Pratama",
    "cimanggu": "Pratama", "cimareme": "Pratama", "gadobangkong": "Pratama",
    "margajaya": "Pratama", "mekarsari": "Pratama", "ngamprah": "Pratama",
    "pakuhaji": "Pratama", "sukatani": "Pratama", "tanimulya": "Pratama",
    "cihanjuangrahayu": "Pratama", "cigugurgirang": "Pratama",
    "cihanjuang": "Pratama", "cihideung": "Pratama", "ciwaruga": "Pratama",
    "karyawangi": "Pratama", "sariwangi": "Pratama", "baranangsiang": "Pratama",
    "cibenda": "Pratama", "cicangkanghilir": "Pratama", "cijambu": "Pratama",
    "cintaasih": "Madya", "karangsari": "Pratama", "bojongsalam": "Pratama",
    "cibedug": "Pratama", "bojonghaleuang": "Pratama", "cikande": "Pratama",
    "cipangeran": "Pratama", "girimukti": "Pratama", "jati": "Pratama",
    "saguling": "Pratama", "citatah": "Pratama", "nyalindung": "Pratama",
    "sumurbandung": "Pratama", "kanangasari": "Pratama", "selacau": "Pratama",
    "kertajaya": "Pratama", "padalarang": "Pratama", "cipatik": "Pratama",
    "gununghalu": "Pratama", "singajaya": "Pratama", "ciharashas": "Pratama",
}

DEFAULT_DESTANA_INDICATORS = [
    "Kelembagaan/kelompok Destana tersedia dan aktif",
    "Perencanaan penanggulangan bencana tersedia",
    "Kajian/informasi risiko bencana tersedia",
    "Sistem peringatan dini dan prosedur evakuasi tersedia",
    "Sosialisasi/edukasi kebencanaan dilaksanakan",
    "Simulasi atau latihan kesiapsiagaan dilaksanakan",
    "Relawan/sumber daya kesiapsiagaan tersedia",
    "Logistik/peralatan tanggap bencana tersedia",
]
DEFAULT_DESTANA_THRESHOLD = 80


def _as_dict(value):
    """
    Firebase RTDB otomatis mengubah dict yang semua kuncinya angka
    ('1','2','3'...) menjadi ARRAY saat disimpan, dan SDK Python
    membacanya balik sebagai list. Fungsi ini mengembalikannya ke dict
    supaya pengecekan isinstance(..., dict) tidak membuang datanya.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(i): v for i, v in enumerate(value) if v is not None}
    return {}


def destana_key(kecamatan, desa):
    return f"{clean_name(kecamatan)}__{clean_name(desa)}"


def get_destana_indicators():
    ref = db.reference('destana/indicators')
    data = _as_dict(ref.get())
    if not data:
        data = {}
        for i, nama in enumerate(DEFAULT_DESTANA_INDICATORS, 1):
            data[str(i)] = {
                'nama': nama,
                'aktif': True,
                'created_at': datetime.now().isoformat(),
                'created_by': 'system'
            }
        ref.set(data)
    return data


def get_destana_threshold():
    value = db.reference('destana/settings/threshold').get()
    try:
        value = float(value)
        return max(1, min(100, value))
    except (TypeError, ValueError):
        return DEFAULT_DESTANA_THRESHOLD


def load_destana_snapshot():
    """
    Baca node 'destana' SEKALI (indikator + setting threshold + checklist
    semua desa) -> return (indicators, threshold, checklists).

    Dulu get_destana_record() memanggil Firebase 2x PER DESA (checklist +
    threshold) sehingga membuka /destana atau Dashboard Admin = ~290
    round-trip Firebase berurutan. Dengan snapshot ini cukup 1 round-trip.
    """
    data = db.reference('destana').get()
    if not isinstance(data, dict):
        data = {}

    indicators = _as_dict(data.get('indicators'))
    if not indicators:
        # Belum ada indikator -> pakai fungsi lama yang sekaligus menyemai
        # indikator bawaan (hanya terjadi sekali di database kosong).
        indicators = get_destana_indicators()

    settings = data.get('settings')
    raw_threshold = settings.get('threshold') if isinstance(settings, dict) else None
    try:
        threshold = max(1, min(100, float(raw_threshold)))
    except (TypeError, ValueError):
        threshold = DEFAULT_DESTANA_THRESHOLD

    checklists = _as_dict(data.get('checklists'))

    return indicators, threshold, checklists


def get_destana_record(kecamatan, desa, indicators, checklists=None, threshold=None):
    """
    Hitung status Destana satu desa. Kalau `checklists` & `threshold` diberikan
    (hasil load_destana_snapshot) TIDAK ada panggilan jaringan; kalau tidak,
    perilaku lama (baca Firebase per desa) dipakai -- untuk pemanggilan tunggal.
    """
    key = destana_key(kecamatan, desa)
    if checklists is not None:
        record = checklists.get(key)
    else:
        record = db.reference(f'destana/checklists/{key}').get()
    record = _as_dict(record)
    checked = _as_dict(record.get('checked'))
    active = [k for k, v in indicators.items() if isinstance(v, dict) and v.get('aktif', True)]
    total = len(active)
    terpenuhi = sum(1 for k in active if bool(checked.get(k)))
    persen = (terpenuhi / total * 100) if total else 0
    if threshold is None:
        threshold = get_destana_threshold()
    return {
        'key': key,
        'checked': checked,
        'terpenuhi': terpenuhi,
        'total': total,
        'persen': persen,
        'status': 'Destana' if total and persen >= threshold else 'Belum memenuhi',
        'kategori_2026': DESTANA_KATEGORI_2026.get(clean_name(desa), '-'),
    }


# ========================================================
# ROUTES - HALAMAN PUBLIK
# ========================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/kondisi")
def kondisi():

    # HTML Folium diambil dari cache sehingga perpindahan halaman tidak
    # memaksa server membaca GeoJSON + membangun seluruh peta lagi.
    peta_html = get_cached_map()

    response = render_template(
        "kondisi.html",
        peta_html=peta_html,
        data_dict=desa_dict,
        kec_dict=kec_dict
    )

    # Browser boleh menyimpan halaman sebentar. Jika data berubah,
    # invalidate_map_cache() dipanggil oleh route mutasi terkait.
    from flask import make_response
    resp = make_response(response)
    resp.headers["Cache-Control"] = "private, max-age=15, must-revalidate"
    return resp


@app.route("/edukasi")
def edukasi():
    return render_template("edukasi.html")


@app.route("/cuaca")
def cuaca():
    return render_template("cuaca.html")


# ========================================================
# ROUTES - AUTH (LOGIN / LOGOUT)
# ========================================================

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_input = request.form['email'].strip()
        password_input = request.form['password']

        # 1. Ambil seluruh data dari node 'users'
        ref = db.reference('users')
        users_data = ref.get()  # Mengembalikan dictionary dari Firebase

        user_found = None

        if users_data:
            # Loop untuk mencari email yang cocok
            for user_id, info in users_data.items():
                if isinstance(info, dict) and info.get('email') == email_input:
                    user_found = info
                    break

        # 2. Cek Password & Role
        if user_found:
            if user_found.get('password') == password_input:
                session['email'] = email_input
                session['role'] = user_found.get('role', 'petugas')

                if session['role'] == 'admin':
                    return redirect(url_for('dashboard_admin'))
                else:
                    return redirect(url_for('dashboard_petugas'))
            else:
                flash("Password salah!")
                return redirect(url_for('login'))
        else:
            flash("Email tidak ditemukan!")
            return redirect(url_for('login'))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Anda telah keluar.")
    return redirect(url_for('home'))



@app.route("/destana")
@login_required
def destana():
    indicators, threshold, checklists = load_destana_snapshot()

    daftar = []
    for key, data in desa_dict.items():
        if not isinstance(data, dict):
            continue
        kec = data.get('Kecamatan', '')
        desa = data.get('Desa', '')
        if not desa:
            continue
        item = get_destana_record(kec, desa, indicators, checklists, threshold)
        item.update({'kecamatan': kec, 'desa': desa})
        daftar.append(item)

    daftar.sort(key=lambda x: (str(x['kecamatan']), str(x['desa'])))
    return render_template(
        "destana.html",
        daftar=daftar,
        indicators=indicators,
        threshold=threshold,
        is_admin=session.get('role') == 'admin',
    )


@app.route("/destana/checklist/<path:desa_key>", methods=['POST'])
@role_required('admin')
def destana_update_checklist(desa_key):
    if not validate_csrf():
        return "CSRF validation failed", 400

    # desa_key dikirim dari server dalam format kecamatan__desa.
    if '__' not in desa_key:
        flash("Identitas daerah tidak valid.")
        return redirect(url_for('destana'))

    kec_clean, desa_clean = desa_key.split('__', 1)
    target = None
    for key, data in desa_dict.items():
        if clean_name(data.get('Kecamatan', '')) == kec_clean and clean_name(data.get('Desa', '')) == desa_clean:
            target = data
            break
    if target is None:
        flash("Daerah tidak ditemukan.")
        return redirect(url_for('destana'))

    indicators = get_destana_indicators()
    checked = {}
    for indicator_id, info in indicators.items():
        if isinstance(info, dict) and info.get('aktif', True):
            checked[indicator_id] = request.form.get(f'indicator_{indicator_id}') == '1'

    db.reference(f'destana/checklists/{desa_key}').set({
        'checked': checked,
        'updated_at': datetime.now().isoformat(),
        'updated_by': session.get('email', '-')
    })
    invalidate_map_cache()
    flash(f"Checklist Destana {target.get('Desa', '-')} berhasil diperbarui.")
    return redirect(url_for('destana'))


@app.route("/admin/destana/indikator/tambah", methods=['POST'])
@role_required('admin')
def destana_tambah_indikator():
    if not validate_csrf():
        return "CSRF validation failed", 400

    nama = request.form.get('nama', '').strip()
    if not nama or len(nama) > 300:
        flash("Nama indikator wajib diisi dan maksimal 300 karakter.")
        return redirect(url_for('destana'))

    ref = db.reference('destana/indicators')
    new_ref = ref.push({
        'nama': nama,
        'aktif': True,
        'created_at': datetime.now().isoformat(),
        'created_by': session.get('email', '-')
    })
    invalidate_map_cache()
    flash("Indikator Destana berhasil ditambahkan.")
    return redirect(url_for('destana'))


@app.route("/admin/destana/indikator/<indicator_id>/edit", methods=['POST'])
@role_required('admin')
def destana_edit_indikator(indicator_id):
    if not validate_csrf():
        return "CSRF validation failed", 400

    nama = request.form.get('nama', '').strip()
    aktif = request.form.get('aktif') == '1'
    if not nama or len(nama) > 300:
        flash("Nama indikator tidak valid.")
        return redirect(url_for('destana'))

    ref = db.reference(f'destana/indicators/{indicator_id}')
    if ref.get() is None:
        flash("Indikator tidak ditemukan.")
        return redirect(url_for('destana'))

    ref.update({
        'nama': nama,
        'aktif': aktif,
        'updated_at': datetime.now().isoformat(),
        'updated_by': session.get('email', '-')
    })
    invalidate_map_cache()
    flash("Indikator Destana berhasil diperbarui.")
    return redirect(url_for('destana'))


@app.route("/admin/destana/indikator/<indicator_id>/hapus", methods=['POST'])
@role_required('admin')
def destana_hapus_indikator(indicator_id):
    if not validate_csrf():
        return "CSRF validation failed", 400

    ref = db.reference(f'destana/indicators/{indicator_id}')
    if ref.get() is None:
        flash("Indikator tidak ditemukan.")
        return redirect(url_for('destana'))

    ref.delete()
    invalidate_map_cache()
    flash("Indikator Destana berhasil dihapus.")
    return redirect(url_for('destana'))


@app.route("/admin/destana/settings", methods=['POST'])
@role_required('admin')
def destana_settings():
    if not validate_csrf():
        return "CSRF validation failed", 400

    try:
        threshold = float(request.form.get('threshold', DEFAULT_DESTANA_THRESHOLD))
        if not 1 <= threshold <= 100:
            raise ValueError
    except (TypeError, ValueError):
        flash("Batas otomatis harus berupa angka 1-100.")
        return redirect(url_for('destana'))

    db.reference('destana/settings').update({
        'threshold': threshold,
        'updated_at': datetime.now().isoformat(),
        'updated_by': session.get('email', '-')
    })
    invalidate_map_cache()
    flash(f"Batas otomatis Destana disimpan: {threshold:g}%.")
    return redirect(url_for('destana'))


# ========================================================
# AI KNOWLEDGE BASE / RAG (ADMIN ONLY)
# ========================================================

@app.route("/admin/ai/knowledge/upload-url", methods=["POST"])
@role_required('admin')
def admin_ai_create_upload_url():
    """Membuat signed upload URL. File tidak melewati Vercel Function."""
    if not validate_csrf():
        return jsonify({"error": "CSRF validation failed"}), 400

    payload = request.get_json(silent=True) or {}
    filename = (payload.get("filename") or "").strip()
    file_size = int(payload.get("file_size") or 0)
    content_type = (payload.get("content_type") or "application/octet-stream").strip()

    allowed = {"pdf", "docx", "txt", "md"}
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if not filename or ext not in allowed:
        return jsonify({"error": "Format yang didukung: PDF, DOCX, TXT, dan MD."}), 400
    if file_size <= 0 or file_size > SUPABASE_MAX_UPLOAD_BYTES:
        return jsonify({"error": "Ukuran dokumen maksimal 50 MB."}), 400

    # Nama file dibuat unik agar dokumen lama tidak tertimpa.
    import uuid
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:180]
    storage_path = f"guides/{datetime.utcnow().strftime('%Y%m%d')}/{uuid.uuid4().hex}_{safe_name}"

    try:
        supabase = get_supabase_admin()
        response = (
            supabase.storage
            .from_(SUPABASE_BUCKET)
            .create_signed_upload_url(storage_path)
        )

        # supabase-py dapat mengembalikan hasil sebagai dict maupun
        # object/APIResponse, tergantung versi storage3 yang terpasang.
        # Ambil token secara kompatibel agar tidak gagal setelah Supabase
        # sebenarnya sudah berhasil membuat signed upload URL.
        data = getattr(response, "data", response)

        if data is None:
            data = {}

        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "dict") and callable(data.dict):
            data = data.dict()
        elif not isinstance(data, dict):
            try:
                data = vars(data)
            except TypeError:
                data = {}

        token = data.get("token") or data.get("signed_upload_token")
        signed_url = (
            data.get("signed_url")
            or data.get("signedUrl")
            or data.get("url")
        )

        if not token:
            # Beberapa versi SDK mengembalikan URL yang token-nya sudah
            # berada di query string. Ekstrak sebagai fallback.
            if signed_url:
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(str(signed_url))
                token_values = parse_qs(parsed.query).get("token") or []
                if token_values:
                    token = token_values[0]

        if not token:
            print(
                "[SUPABASE] create_signed_upload_url response:",
                repr(response),
                "data:",
                repr(data),
            )
            raise RuntimeError(
                "Supabase berhasil merespons tetapi token upload tidak ditemukan."
            )

        return jsonify({
            "path": storage_path,
            "token": token,
            "filename": filename,
            "content_type": content_type,
            "bucket": SUPABASE_BUCKET,
        })
    except Exception as exc:
        print(f"[SUPABASE] create signed upload URL gagal: {type(exc).__name__}: {exc}")
        return jsonify({"error": "Gagal menyiapkan upload dokumen ke Supabase."}), 500


@app.route("/admin/ai/knowledge/process", methods=["POST"])
@role_required('admin')
def admin_ai_process_uploaded():
    """Dipanggil setelah browser selesai direct-upload ke Supabase Storage."""
    if not validate_csrf():
        return jsonify({"error": "CSRF validation failed"}), 400

    payload = request.get_json(silent=True) or {}
    storage_path = (payload.get("path") or "").strip()
    filename = (payload.get("filename") or "").strip()
    file_size = int(payload.get("file_size") or 0)

    if not storage_path.startswith("guides/") or not filename:
        return jsonify({"error": "Referensi dokumen tidak valid."}), 400
    if file_size <= 0 or file_size > SUPABASE_MAX_UPLOAD_BYTES:
        return jsonify({"error": "Ukuran dokumen tidak valid."}), 400

    try:
        supabase = get_supabase_admin()
        downloaded = supabase.storage.from_(SUPABASE_BUCKET).download(storage_path)
        raw = getattr(downloaded, "data", downloaded)
        if not isinstance(raw, (bytes, bytearray)):
            raise RuntimeError("Supabase tidak mengembalikan isi file.")

        text, sha256 = extract_text_bytes(filename, bytes(raw))
        doc_id, chunks = save_document(
            filename,
            text,
            sha256,
            session.get("email", "-"),
            storage_path=storage_path,
            file_size=file_size,
        )

        return jsonify({
            "ok": True,
            "document_id": doc_id,
            "chunks": chunks,
            "message": f"Panduan berhasil dipelajari: {filename} ({chunks} bagian).",
        })
    except Exception as exc:
        print(f"[SUPABASE/RAG] processing gagal: {type(exc).__name__}: {exc}")
        # Jangan tinggalkan file yatim jika pemrosesan gagal.
        try:
            get_supabase_admin().storage.from_(SUPABASE_BUCKET).remove([storage_path])
        except Exception:
            pass
        return jsonify({"error": f"Gagal memproses panduan: {exc}"}), 500


@app.route("/admin/ai/knowledge/delete", methods=["POST"])
@role_required('admin')
def admin_ai_delete():
    if not validate_csrf():
        return "CSRF validation failed", 400
    doc_id = (request.form.get("document_id") or "").strip()
    try:
        delete_document(doc_id)
        flash("Dokumen dan pengetahuan terkait berhasil dihapus.")
    except Exception as exc:
        flash(f"Gagal menghapus dokumen: {exc}")
    return redirect(url_for('dashboard_admin'))


@app.route("/admin/ai/knowledge/text", methods=["POST"])
@role_required('admin')
def admin_ai_add_text():
    if not validate_csrf():
        return "CSRF validation failed", 400
    try:
        title = request.form.get('title', '').strip()
        text = request.form.get('content', '').strip()
        doc_id, chunks = save_manual_knowledge(title, text, session.get('email', '-'))
        flash(f"Pengetahuan AI berhasil ditambahkan ({chunks} bagian).")
    except Exception as e:
        flash(f"Gagal menyimpan pengetahuan AI: {e}")
    return redirect(url_for('dashboard_admin'))


# ========================================================
# ROUTES - DASHBOARD ADMIN
# ========================================================

@app.route("/dashboard-admin")
@role_required('admin')
def dashboard_admin():

    # Semua bacaan jaringan di bawah saling bebas, jadi dijalankan SERENTAK
    # (waktu total = yang paling lambat, bukan dijumlahkan satu per satu).
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="admin-io") as ex:
        f_users = ex.submit(lambda: db.reference('users').get())
        f_laporan = ex.submit(lambda: db.reference('laporan_edukasi').get())
        f_dokumen = ex.submit(list_documents)

        # --- Daftar akun (users) ---
        users_data = f_users.result() or {}

        # --- Daftar laporan edukasi dari semua petugas ---
        laporan_data = f_laporan.result() or {}
        try:
            ai_knowledge_documents = f_dokumen.result()
        except Exception as exc:
            print(f"[SUPABASE] list knowledge gagal: {type(exc).__name__}: {exc}")
            ai_knowledge_documents = []

    laporan_list = []
    for laporan_id, info in laporan_data.items():
        if isinstance(info, dict):
            info = dict(info)
            info['id'] = laporan_id
            laporan_list.append(info)
    laporan_list.sort(key=lambda x: x.get('tanggal', ''), reverse=True)

    total_orang_teredukasi_laporan = sum(
        l.get('jumlah_orang_teredukasi', 0) for l in laporan_list
    )

    # --- Daftar desa untuk panel "Kelola Data Desa" ---
    desa_list = sorted(
        desa_dict.items(),
        key=lambda item: (item[1].get('Kecamatan', ''), item[1].get('Desa', ''))
    )

    # --- Ringkasan statistik untuk kartu dashboard ---
    total_kecamatan = len(kec_dict)
    total_desa = len(desa_dict)
    total_warga_wajib_edukasi = sum(
        d.get('Warga_Wajib_Edukasi_Desa', 0) for d in desa_dict.values()
    )
    total_rentan = sum(
        d.get('Total_Rentan_Desa', 0) for d in desa_dict.values()
    )

    # --- Persentase realisasi edukasi keseluruhan KBB (real, dari laporan) ---
    persen_realisasi_keseluruhan = round(
        min(100, (total_orang_teredukasi_laporan / total_warga_wajib_edukasi * 100)), 2
    ) if total_warga_wajib_edukasi > 0 else 0.0

    kec_list = sorted(kec_dict.items(), key=lambda item: item[1].get('Kecamatan', ''))

    # --- Data desa & kecamatan yang BELUM tercatat (dihitung saat startup) ---
    desa_kosong_list = DESA_KOSONG_LIST
    kec_kosong_list = KEC_KOSONG_LIST
    total_desa_kosong = len(desa_kosong_list)
    total_kec_kosong = len(kec_kosong_list)

    return render_template(
        "Dashboard_admin.html",
        kajian_ringkasan=kajian_risiko.ringkasan_matriks(MATRIKS_KAJIAN),
        kajian_wilayah=_wilayah_kajian(),
        kajian_hazards=kajian_risiko.HAZARDS,
        users=users_data,
        laporan_list=laporan_list,
        total_orang_teredukasi_laporan=total_orang_teredukasi_laporan,
        desa_list=desa_list,
        kec_list=kec_list,
        total_kecamatan=total_kecamatan,
        total_desa=total_desa,
        total_warga_wajib_edukasi=total_warga_wajib_edukasi,
        total_rentan=total_rentan,
        persen_realisasi_keseluruhan=persen_realisasi_keseluruhan,
        desa_kosong_list=desa_kosong_list,
        kec_kosong_list=kec_kosong_list,
        total_desa_kosong=total_desa_kosong,
        total_kec_kosong=total_kec_kosong,
        ai_knowledge_documents=ai_knowledge_documents,
        supabase_url=SUPABASE_URL,
        supabase_publishable_key=SUPABASE_PUBLISHABLE_KEY,
        supabase_bucket=SUPABASE_BUCKET,
    )


# --- Kelola Akun ---

@app.route("/admin/akun/tambah", methods=['POST'])
@role_required('admin')
def admin_tambah_akun():
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()
    role = request.form.get('role', 'petugas')

    if not email or not password:
        flash("Email dan password wajib diisi.")
        return redirect(url_for('dashboard_admin'))

    ref = db.reference('users')
    existing = ref.get() or {}
    for info in existing.values():
        if isinstance(info, dict) and info.get('email') == email:
            flash("Email tersebut sudah terdaftar.")
            return redirect(url_for('dashboard_admin'))

    ref.push({'email': email, 'password': password, 'role': role})
    flash(f"Akun {email} berhasil ditambahkan sebagai {role}.")
    return redirect(url_for('dashboard_admin'))


@app.route("/admin/akun/<user_id>/edit", methods=['POST'])
@role_required('admin')
def admin_edit_akun(user_id):
    role = request.form.get('role', 'petugas')
    ref = db.reference(f'users/{user_id}')

    if ref.get() is None:
        flash("Akun tidak ditemukan.")
        return redirect(url_for('dashboard_admin'))

    ref.update({'role': role})
    flash("Role akun berhasil diperbarui.")
    return redirect(url_for('dashboard_admin'))


@app.route("/admin/akun/<user_id>/hapus", methods=['POST'])
@role_required('admin')
def admin_hapus_akun(user_id):
    ref = db.reference(f'users/{user_id}')
    user_data = ref.get()

    if user_data is None:
        flash("Akun tidak ditemukan.")
        return redirect(url_for('dashboard_admin'))

    if user_data.get('email') == session.get('email'):
        flash("Tidak bisa menghapus akun yang sedang Anda gunakan untuk login.")
        return redirect(url_for('dashboard_admin'))

    ref.delete()
    flash("Akun berhasil dihapus.")
    return redirect(url_for('dashboard_admin'))


# --- Kelola Data Desa ---

@app.route("/admin/edit-desa/<desa_key>", methods=['POST'])
@role_required('admin')
def admin_edit_desa(desa_key):
    if desa_key not in desa_dict:
        flash("Data desa tidak ditemukan.")
        return redirect(url_for('dashboard_admin'))

    warga = safe_number(request.form.get('warga_wajib_edukasi', 0))
    bl = safe_number(request.form.get('rentan_balita_lansia', 0))
    miskin = safe_number(request.form.get('rentan_miskin', 0))
    disabilitas = safe_number(request.form.get('rentan_disabilitas', 0))

    terapkan_data_desa(desa_key, warga, bl, miskin, disabilitas)

    data = desa_dict[desa_key]

    # Target warga wajib edukasi desa ini berubah -> persentase realisasi
    # edukasi (real) juga harus dihitung ulang (jumlah aktual yang sudah
    # diedukasi dari laporan petugas tidak berubah, tapi pembaginya beda).
    aktual_desa = data.get('Jumlah_Teredukasi_Aktual_Desa', 0)
    data['Persen_Realisasi_Edukasi_Desa'] = round(
        min(100, (aktual_desa / warga * 100)), 2
    ) if warga > 0 else 0.0

    # Kecamatan induk juga perlu disinkronkan ulang (baik target warga
    # wajib edukasi maupun realisasi aktualnya), karena keduanya adalah
    # agregat dari seluruh desa di kecamatan tsb.
    key_kec = clean_name(data.get('Kecamatan', ''))
    if key_kec in kec_dict:
        desa_sekecamatan = [
            dd for dd in desa_dict.values()
            if clean_name(dd.get('Kecamatan', '')) == key_kec
        ]
        total_target_kec = sum(
            dd.get('Warga_Wajib_Edukasi_Desa', 0) for dd in desa_sekecamatan
        )
        total_aktual_kec = sum(
            dd.get('Jumlah_Teredukasi_Aktual_Desa', 0) for dd in desa_sekecamatan
        )

        k = kec_dict[key_kec]
        k['Total_Warga_Wajib_Edukasi_Kecamatan'] = total_target_kec
        k['Jumlah_Teredukasi_Aktual_Kec'] = total_aktual_kec
        k['Persen_Realisasi_Edukasi_Kec'] = round(
            min(100, (total_aktual_kec / total_target_kec * 100)), 2
        ) if total_target_kec > 0 else 0.0

    # --- Simpan permanen ke Firebase (bukan ke file Excel) ---
    # File Excel di server TIDAK BISA ditulis di hosting serverless
    # (Vercel: read-only filesystem). Firebase adalah sumber kebenaran
    # untuk perubahan admin; nilai ini diterapkan ulang ke desa_dict
    # setiap kali server start lewat sync_desa_overrides().
    try:
        db.reference(f'desa_overrides/{desa_key}').set({
            'warga': warga,
            'bl': bl,
            'miskin': miskin,
            'disabilitas': disabilitas,
            'updated_at': datetime.now().isoformat(),
            'updated_by': session.get('email', '-'),
        })
    except Exception as e:
        flash(f"Data di layar sudah update, tapi GAGAL disimpan permanen ke Firebase: {e}")
        return redirect(url_for('dashboard_admin'))

    # Best-effort: kalau kebetulan jalan di lingkungan lokal (filesystem
    # bisa ditulis), sinkronkan juga ke file Excel biar konsisten kalau
    # dibuka manual. Boleh gagal (mis. di Vercel) tanpa mematikan request.
    try:
        update_desa_excel(
            app.root_path,
            data.get('Desa', ''),
            data.get('Kecamatan', ''),
            warga, bl, miskin, disabilitas
        )
    except Exception:
        pass

    invalidate_map_cache()
    flash(f"Data desa {data.get('Desa', '-')} berhasil diperbarui.")
    return redirect(url_for('dashboard_admin'))


# --- Kelola Kajian Risiko Bencana (unggah Excel / isi manual) ---
#
# Data = format file matriks BPBD (sheet DESA/KECAMATAN/KABUPATEN). Lihat
# kajian_risiko.py. Disimpan di Firebase node 'kajian_risiko' (folder deploy
# Vercel read-only) dan dipakai peta tab "Kajian Risiko Bencana".

_KAJIAN_HASH = '#kajian-risiko'  # supaya setelah simpan, tab yang sama terbuka lagi


def _kembali_ke_tab_kajian():
    return redirect(url_for('dashboard_admin') + _KAJIAN_HASH)


@lru_cache(maxsize=1)
def _wilayah_kajian():
    """{nama kecamatan: [nama desa, ...]} dari GeoJSON peta (nama yang sama dgn peta)."""
    hasil = {}
    for f in load_local_geojson_files().get("features", []):
        props = f.get("properties") or {}
        kec = str(props.get("nm_kecamatan", "")).strip()
        desa = get_nama_desa(props).strip()
        if not kec or not desa or kajian_risiko.key_js(desa) == "waduk":
            continue
        hasil.setdefault(kec, set()).add(desa)
    return {k: sorted(v) for k, v in sorted(hasil.items())}


def _cek_wilayah_kajian(level, nama_kec, nama_desa):
    wilayah = _wilayah_kajian()
    if level in ("kec", "desa") and nama_kec not in wilayah:
        raise ValueError("Kecamatan tidak dikenal di peta.")
    if level == "desa" and nama_desa not in wilayah[nama_kec]:
        raise ValueError("Desa tidak ditemukan di kecamatan tsb.")


def _laporan_kecocokan_kajian(m):
    """Nama wilayah di data yang tidak ketemu di peta, dan sebaliknya."""
    fitur_desa = load_local_geojson_files().get("features", [])
    fitur_kec = [{"properties": {"nm_kecamatan": n}} for n in _wilayah_kajian()]
    return kajian_risiko.siapkan_data_peta(m, fitur_desa, fitur_kec, get_nama_desa)["laporan"]


@app.route("/admin/kajian-risiko/upload", methods=['POST'])
@role_required('admin')
def admin_kajian_upload():
    if not validate_csrf():
        return "CSRF validation failed", 400

    f = request.files.get('file_matriks')
    if not f or not f.filename:
        flash("❌ Pilih file Excel (.xlsx) dulu sebelum mengunggah.")
        return _kembali_ke_tab_kajian()

    try:
        m = kajian_risiko.muat_matriks_dari_bytes(
            f.read(kajian_risiko.MAX_UKURAN_UNGGAH + 1), f.filename
        )
    except ValueError as e:
        flash(f"❌ Unggahan ditolak: {e}")
        return _kembali_ke_tab_kajian()

    meta = {
        'file': m['file'],
        'sumber': 'upload',
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'updated_by': session.get('email', '-'),
        'version': int(time.time() * 1000),
    }
    try:
        db.reference('kajian_risiko').set(kajian_risiko.matriks_ke_dict(m, meta))
    except Exception as e:
        flash(f"❌ File terbaca, tapi GAGAL disimpan ke Firebase (data lama tetap dipakai): {e}")
        return _kembali_ke_tab_kajian()

    m['meta'] = meta
    with _KAJIAN_LOCK:
        _pasang_matriks_kajian(m, True)
    invalidate_map_cache()

    pesan = (
        f"Kajian Risiko berhasil diganti dengan {m['file']}: "
        f"{len(m['desa'])} desa, {len(m['kec'])} kecamatan, {len(m['kab'])} baris kabupaten."
    )
    try:
        lap = _laporan_kecocokan_kajian(m)
        if lap['matriks_tak_terpakai']:
            pesan += (
                f" ⚠️ {len(lap['matriks_tak_terpakai'])} desa di Excel tidak ketemu di peta "
                f"(cek ejaan): {'; '.join(lap['matriks_tak_terpakai'][:5])}"
                + (" ..." if len(lap['matriks_tak_terpakai']) > 5 else "")
            )
        if lap['desa_tanpa_data']:
            pesan += f" ⚠️ {len(lap['desa_tanpa_data'])} desa di peta belum punya data."
    except Exception as e:  # noqa: BLE001 - laporan hanya tambahan
        print(f"⚠️  Laporan kecocokan Kajian Risiko gagal: {e}")
    if m['warnings']:
        pesan += f" ({len(m['warnings'])} catatan dari pembacaan file, lihat kartu status.)"
    flash(pesan)
    return _kembali_ke_tab_kajian()


def _simpan_record_kajian(level, kk, dk, hz, rec):
    """
    Terapkan 1 perubahan (rec=None -> hapus) ke memori lalu simpan ke Firebase.
    Kalau data aktif masih berasal dari file bawaan, SELURUH matriks dipindah
    ke Firebase dulu (supaya node Firebase tidak berisi 1 baris saja lalu
    menimpa data bawaan saat server restart). Gagal simpan -> memori dikembalikan.
    """
    global MATRIKS_KAJIAN
    with _KAJIAN_LOCK:
        m = MATRIKS_KAJIAN or {
            "file": "isian manual", "desa": {}, "kec": {}, "kab": {}, "warnings": [],
        }
        lama = kajian_risiko.cari_record(m, level, kk, dk, hz)
        lama = dict(lama) if lama else None
        kajian_risiko.terapkan_record(m, level, kk, dk, hz, rec)

        versi = int(time.time() * 1000)
        meta = dict(m.get('meta') or {})
        meta.update({
            'file': m.get('file', '-'),
            'sumber': 'manual',
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'updated_by': session.get('email', '-'),
            'version': versi,
        })
        try:
            ref = db.reference('kajian_risiko')
            if _KAJIAN_DI_FIREBASE:
                ref.update({
                    kajian_risiko.path_firebase_record(level, kk, dk, hz): rec,
                    'meta/sumber': meta['sumber'],
                    'meta/updated_at': meta['updated_at'],
                    'meta/updated_by': meta['updated_by'],
                    'meta/version': versi,
                })
            else:
                ref.set(kajian_risiko.matriks_ke_dict(m, meta))
        except Exception:
            kajian_risiko.terapkan_record(m, level, kk, dk, hz, lama)
            raise

        m['meta'] = meta
        _pasang_matriks_kajian(m, True)
    invalidate_map_cache()


@app.route("/admin/kajian-risiko/simpan", methods=['POST'])
@role_required('admin')
def admin_kajian_simpan():
    if not validate_csrf():
        return "CSRF validation failed", 400
    try:
        level, kk, dk, hz, rec = kajian_risiko.buat_record_manual(request.form)
        _cek_wilayah_kajian(level, rec.get('kec', ''), rec.get('desa', ''))
    except ValueError as e:
        flash(f"❌ {e}")
        return _kembali_ke_tab_kajian()

    try:
        _simpan_record_kajian(level, kk, dk, hz, rec)
    except Exception as e:
        flash(f"❌ Gagal menyimpan ke Firebase, perubahan dibatalkan: {e}")
        return _kembali_ke_tab_kajian()

    label = kajian_risiko.HAZARD_BY_ID[hz]['label']
    lokasi = rec.get('desa') or rec.get('kec') or 'Kabupaten Bandung Barat'
    flash(f"Data kajian risiko {label} untuk {lokasi} disimpan (Kelas Risiko: {rec['r']}).")
    return _kembali_ke_tab_kajian()


@app.route("/admin/kajian-risiko/hapus", methods=['POST'])
@role_required('admin')
def admin_kajian_hapus():
    if not validate_csrf():
        return "CSRF validation failed", 400
    try:
        level, kk, dk, hz, nama_kec, nama_desa = kajian_risiko.kunci_dari_form(request.form)
        _cek_wilayah_kajian(level, nama_kec, nama_desa)
    except ValueError as e:
        flash(f"❌ {e}")
        return _kembali_ke_tab_kajian()

    if kajian_risiko.cari_record(MATRIKS_KAJIAN, level, kk, dk, hz) is None:
        flash("❌ Data yang mau dihapus tidak ditemukan.")
        return _kembali_ke_tab_kajian()

    try:
        _simpan_record_kajian(level, kk, dk, hz, None)
    except Exception as e:
        flash(f"❌ Gagal menghapus di Firebase, perubahan dibatalkan: {e}")
        return _kembali_ke_tab_kajian()

    flash(f"Data kajian risiko {kajian_risiko.HAZARD_BY_ID[hz]['label']} untuk "
          f"{nama_desa or nama_kec or 'Kabupaten Bandung Barat'} dihapus.")
    return _kembali_ke_tab_kajian()


@app.route("/admin/kajian-risiko/data")
@role_required('admin')
def admin_kajian_data():
    """Isi awal form manual: nilai yang sudah tersimpan untuk wilayah + bencana terpilih."""
    try:
        level, kk, dk, hz, nama_kec, nama_desa = kajian_risiko.kunci_dari_form(request.args)
    except ValueError as e:
        return jsonify({"ada": False, "error": str(e)}), 400
    rec = kajian_risiko.cari_record(MATRIKS_KAJIAN, level, kk, dk, hz)
    if not rec:
        return jsonify({"ada": False})
    return jsonify({"ada": True, "record": {
        "kelas_risiko": rec.get("r"), "kelas_bahaya": rec.get("b"),
        "kelas_kerentanan": rec.get("v"), "kelas_kapasitas": rec.get("c"),
        "penduduk": rec.get("p"), "kerugian": rec.get("k"), "luas_risiko": rec.get("lr"),
    }})


@app.route("/admin/kajian-risiko/template")
@role_required('admin')
def admin_kajian_template():
    from flask import send_file
    path = os.path.join(app.root_path, 'data', 'template', 'TEMPLATE_KAJIAN_RISIKO_BENCANA.xlsx')
    return send_file(path, as_attachment=True, download_name='TEMPLATE_KAJIAN_RISIKO_BENCANA.xlsx')


# ========================================================
# ROUTES - DASHBOARD PETUGAS
# ========================================================

_QUERY_LAPORAN_PETUGAS_OK = True


def _ambil_laporan_petugas(email):
    """
    Ambil laporan milik SATU petugas. Filter dilakukan di server Firebase
    (order_by_child + equal_to) sehingga yang diunduh hanya laporan petugas
    itu -- bukan seluruh node 'laporan_edukasi' yang terus membesar (laporan
    "seluruh KBB" menambah ~150 baris sekali kirim).

    Butuh aturan indeks di Firebase Realtime Database:
        "laporan_edukasi": { ".indexOn": ["petugas_email"] }
    Kalau indeks belum ada, Firebase menolak query -> otomatis kembali ke cara
    lama (unduh semua lalu filter di Python) dan tidak dicoba lagi sampai
    server restart, supaya tidak menambah 1 request gagal di tiap halaman.
    """
    global _QUERY_LAPORAN_PETUGAS_OK
    if _QUERY_LAPORAN_PETUGAS_OK and email:
        try:
            return (
                db.reference('laporan_edukasi')
                .order_by_child('petugas_email')
                .equal_to(email)
                .get()
            ) or {}
        except Exception as exc:
            _QUERY_LAPORAN_PETUGAS_OK = False
            print(
                "⚠️  Query laporan per petugas gagal (kemungkinan .indexOn "
                f"'petugas_email' belum ada di Firebase Rules): {exc}"
            )
    semua = db.reference('laporan_edukasi').get() or {}
    return {
        k: v for k, v in semua.items()
        if isinstance(v, dict) and v.get('petugas_email') == email
    }


@app.route("/dashboard-petugas")
@role_required('petugas', 'admin')
def dashboard_petugas():

    laporan_data = _ambil_laporan_petugas(session.get('email'))
    laporan_list = []
    for laporan_id, info in laporan_data.items():
        if isinstance(info, dict) and info.get('petugas_email') == session.get('email'):
            info = dict(info)
            info['id'] = laporan_id
            laporan_list.append(info)
    laporan_list.sort(key=lambda x: x.get('tanggal', ''), reverse=True)

    total_orang_teredukasi_saya = sum(
        l.get('jumlah_orang_teredukasi', 0) for l in laporan_list
    )

    # Susun daftar kecamatan -> desa untuk dropdown form laporan
    desa_by_kecamatan = {}
    for d in desa_dict.values():
        kec_nama = d.get('Kecamatan', '-')
        desa_by_kecamatan.setdefault(kec_nama, [])
        if d.get('Desa') not in desa_by_kecamatan[kec_nama]:
            desa_by_kecamatan[kec_nama].append(d.get('Desa'))
    for kec_nama in desa_by_kecamatan:
        desa_by_kecamatan[kec_nama].sort()

    return render_template(
        "Dashboard_petugas.html",
        laporan_list=laporan_list,
        total_orang_teredukasi_saya=total_orang_teredukasi_saya,
        desa_by_kecamatan=desa_by_kecamatan,
    )


@app.route("/petugas/lapor", methods=['POST'])
@role_required('petugas', 'admin')
def petugas_lapor():
    cakupan = request.form.get('cakupan', 'desa').strip()
    deskripsi = request.form.get('deskripsi', '').strip()
    jumlah_kk = int(safe_number(request.form.get('jumlah_kk', 0)))

    if not deskripsi or jumlah_kk <= 0:
        flash("Jumlah KK yang diedukasi dan deskripsi kegiatan wajib diisi dengan benar.")
        return redirect(url_for('dashboard_petugas'))

    # ----------------------------------------------------------
    # CAKUPAN "SELURUH KBB"
    # ----------------------------------------------------------
    # Dipakai kalau acara edukasinya mengundang lintas kecamatan
    # se-Kabupaten Bandung Barat (mis. acara di tingkat kabupaten),
    # jadi petugas TIDAK PUNYA kepastian peserta berasal dari desa
    # mana saja. Daripada dipaksa pilih 1 desa (yang akan bikin data
    # realisasi desa itu bias/menggelembung) atau tidak dicatat sama
    # sekali (data hilang), jumlah KK dialokasikan ke SEMUA desa di
    # desa_dict dengan bobot SAMA RATA (asas tak berbeda / principle
    # of indifference: tanpa info asal peserta, tiap desa dianggap
    # berpeluang sama), lalu dibulatkan dengan Metode Kuota + Sisa
    # Terbesar (lihat bagikan_kuota_sisa_terbesar() di ml_engine.py)
    # supaya totalnya tetap PERSIS sama dengan jumlah_kk yang
    # diinput petugas -- bukan cuma dibagi rata & dibulatkan asal2an.
    #
    # Setiap desa yang kebagian alokasi akan punya BARIS LAPORAN
    # SENDIRI (bukan digabung jadi satu), ditandai
    # 'sumber_estimasi': 'rata_seluruh_kbb' dan berbagi
    # 'laporan_induk_id' yang sama, supaya:
    #   1. Tetap konsisten dengan skema data desa/kecamatan yang
    #      sudah ada (dashboard, peta /kondisi, dsb tidak perlu ubah).
    #   2. Admin/petugas bisa lihat & audit bahwa angka itu HASIL
    #      ESTIMASI, bukan pendataan langsung per desa.
    # ----------------------------------------------------------
    if cakupan == 'kbb':
        # Bobot sama rata untuk semua desa yang ada di sistem.
        bobot_rata = {key_desa: 1 for key_desa in desa_dict.keys()}
        alokasi_kk = bagikan_kuota_sisa_terbesar(jumlah_kk, bobot_rata)

        laporan_induk_id = secrets.token_hex(6)
        ref = db.reference('laporan_edukasi')
        tanggal_sekarang = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_orang = 0
        jumlah_desa_kebagian = 0

        for key_desa, kk_desa in alokasi_kk.items():
            if kk_desa <= 0:
                continue

            d = desa_dict.get(key_desa, {})
            nama_desa = d.get('Desa', '-')
            nama_kec = d.get('Kecamatan', '-')
            orang_desa = kk_desa * 4
            total_orang += orang_desa
            jumlah_desa_kebagian += 1

            ref.push({
                'petugas_email': session.get('email'),
                'kecamatan': nama_kec,
                'desa': nama_desa,
                'jumlah_kk': kk_desa,
                'jumlah_orang_teredukasi': orang_desa,
                'deskripsi': deskripsi,
                'tanggal': tanggal_sekarang,
                'cakupan_asli': 'seluruh_kbb',
                'sumber_estimasi': 'rata_seluruh_kbb',
                'laporan_induk_id': laporan_induk_id,
                'catatan_estimasi': (
                    f"Bagian dari laporan acara se-KBB ({jumlah_kk} KK total). "
                    "Jumlah KK di desa ini adalah HASIL ALOKASI OTOMATIS "
                    "(asumsi rata ke semua desa, dibulatkan dengan Metode "
                    "Kuota Sisa Terbesar), BUKAN pendataan langsung per desa, "
                    "karena peserta acara berasal dari seluruh Kabupaten "
                    "Bandung Barat tanpa data asal per desa."
                ),
            })
            tambah_realisasi_edukasi(nama_kec, nama_desa, orang_desa)

        flash(
            f"Laporan edukasi seluruh KBB berhasil dikirim: {jumlah_kk} KK "
            f"({total_orang} orang) dialokasikan rata ke {jumlah_desa_kebagian} desa "
            "(estimasi, bukan pendataan per desa)."
        )
        return redirect(url_for('dashboard_petugas'))

    # ----------------------------------------------------------
    # CAKUPAN "DESA TERTENTU" (perilaku lama, tidak berubah)
    # ----------------------------------------------------------
    kecamatan = request.form.get('kecamatan', '').strip()
    desa = request.form.get('desa', '').strip()

    if not kecamatan or not desa:
        flash("Kecamatan dan desa wajib diisi kalau cakupan laporan bukan seluruh KBB.")
        return redirect(url_for('dashboard_petugas'))

    # Setiap orang yang datang dianggap mewakili 1 KK.
    # Diasumsikan rata-rata 1 KK = 4 orang, jadi jumlah orang yang
    # sebenarnya tercatat teredukasi = jumlah KK yang hadir x 4.
    jumlah_orang_teredukasi = jumlah_kk * 4

    ref = db.reference('laporan_edukasi')
    ref.push({
        'petugas_email': session.get('email'),
        'kecamatan': kecamatan,
        'desa': desa,
        'jumlah_kk': jumlah_kk,
        'jumlah_orang_teredukasi': jumlah_orang_teredukasi,
        'deskripsi': deskripsi,
        'tanggal': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cakupan_asli': 'desa_tertentu',
    })

    # Update INSTAN persentase realisasi edukasi di memori supaya
    # halaman /kondisi (peta, popup, chatbot) langsung menampilkan
    # angka terbaru begitu laporan ini terkirim, tanpa restart server.
    tambah_realisasi_edukasi(kecamatan, desa, jumlah_orang_teredukasi)

    flash(
        f"Laporan edukasi berhasil dikirim: {jumlah_kk} KK "
        f"({jumlah_orang_teredukasi} orang) di Desa {desa} tercatat teredukasi."
    )
    return redirect(url_for('dashboard_petugas'))


@app.route("/petugas/lapor/edit/<laporan_id>", methods=['POST'])
@role_required('petugas', 'admin')
def petugas_edit_lapor(laporan_id):
    """
    Ubah laporan edukasi yang sudah ada (mis. jumlah KK & deskripsi salah
    input). Hanya petugas pemilik laporan (atau admin) yang boleh mengubah.

    Kecamatan & desa laporan TIDAK diubah di sini (biar sederhana & aman) --
    kalau lokasinya salah, lebih baik dihapus lalu dibuat laporan baru.
    """
    ref = db.reference(f'laporan_edukasi/{laporan_id}')
    laporan = ref.get()

    if not laporan:
        flash("Laporan tidak ditemukan (mungkin sudah dihapus).")
        return redirect(url_for('dashboard_petugas'))

    if laporan.get('petugas_email') != session.get('email') and session.get('role') != 'admin':
        flash("Anda tidak berhak mengubah laporan milik petugas lain.")
        return redirect(url_for('dashboard_petugas'))

    deskripsi_baru = request.form.get('deskripsi', '').strip()
    jumlah_kk_baru = int(safe_number(request.form.get('jumlah_kk', 0)))

    if jumlah_kk_baru <= 0 or not deskripsi_baru:
        flash("Jumlah KK dan deskripsi kegiatan wajib diisi dengan benar.")
        return redirect(url_for('dashboard_petugas'))

    jumlah_orang_lama = laporan.get('jumlah_orang_teredukasi', 0)
    # Sama seperti saat lapor baru: 1 KK yang hadir dianggap 4 orang.
    jumlah_orang_baru = jumlah_kk_baru * 4

    kecamatan = laporan.get('kecamatan', '')
    desa = laporan.get('desa', '')

    ref.update({
        'jumlah_kk': jumlah_kk_baru,
        'jumlah_orang_teredukasi': jumlah_orang_baru,
        'deskripsi': deskripsi_baru,
    })

    # Selisih (bisa negatif kalau KK dikurangi) langsung diterapkan ke
    # realisasi edukasi desa/kecamatan yang sedang berjalan di memori,
    # supaya /kondisi tidak perlu restart server untuk update.
    selisih_orang = jumlah_orang_baru - jumlah_orang_lama
    if selisih_orang != 0:
        tambah_realisasi_edukasi(kecamatan, desa, selisih_orang)

    flash(
        f"Laporan berhasil diperbarui: {jumlah_kk_baru} KK "
        f"({jumlah_orang_baru} orang) di Desa {desa}."
    )
    return redirect(url_for('dashboard_petugas'))


@app.route("/petugas/lapor/hapus/<laporan_id>", methods=['POST'])
@role_required('petugas', 'admin')
def petugas_hapus_lapor(laporan_id):
    """
    Hapus satu laporan edukasi. Hanya petugas pemilik laporan (atau admin)
    yang boleh menghapus.
    """
    ref = db.reference(f'laporan_edukasi/{laporan_id}')
    laporan = ref.get()

    if not laporan:
        flash("Laporan tidak ditemukan (mungkin sudah dihapus sebelumnya).")
        return redirect(url_for('dashboard_petugas'))

    if laporan.get('petugas_email') != session.get('email') and session.get('role') != 'admin':
        flash("Anda tidak berhak menghapus laporan milik petugas lain.")
        return redirect(url_for('dashboard_petugas'))

    kecamatan = laporan.get('kecamatan', '')
    desa = laporan.get('desa', '')
    jumlah_orang = laporan.get('jumlah_orang_teredukasi', 0)

    ref.delete()

    # Kurangi realisasi edukasi di memori sejumlah kontribusi laporan
    # yang baru saja dihapus.
    if jumlah_orang:
        tambah_realisasi_edukasi(kecamatan, desa, -jumlah_orang)

    flash(f"Laporan edukasi ({laporan.get('jumlah_kk', 0)} KK) di Desa {desa} berhasil dihapus.")
    return redirect(url_for('dashboard_petugas'))


# ========================================================
# GLOSARIUM: PENJELASAN ISTILAH & RUMUS
# ========================================================
# Chatbot di /api/chat itu rule-based (lihat catatan di bawah),
# bukan model AI generatif -- jadi dia tidak punya "system prompt"
# yang bisa "dipahamkan" seperti LLM. Sebagai gantinya, di sinilah
# tempatnya nambahin pengetahuan: kalau pesan user kedeteksi
# sedang menanyakan ARTI/RUMUS suatu istilah (bukan menanyakan
# data desa/kecamatan tertentu), chatbot menjawab dari glosarium
# di bawah ini.
# ========================================================

GLOSARIUM_EDUKASI = {
    "prioritas": (
        "**Persentase Prioritas** mengukur seberapa besar *porsi* warga "
        "wajib edukasi di suatu desa/kecamatan dibanding wilayah lain, "
        "dihitung dari data populasi (bukan dari laporan kegiatan). "
        "Rumusnya:\n\n"
        "- *Desa*: Warga Wajib Diedukasi Desa ÷ Warga Terpapar Kecamatan "
        "× 100%\n"
        "- *Kecamatan*: Warga Terpapar Kecamatan ini ÷ Total Warga "
        "Terpapar Seluruh Kecamatan × 100%\n\n"
        "Semakin tinggi angka ini, semakin besar porsi warga wajib "
        "edukasi di wilayah itu dibanding wilayah lain -- inilah yang "
        "jadi dasar warna di peta."
    ),
    "realisasi": (
        "**Persentase Realisasi Edukasi** adalah progres NYATA di "
        "lapangan: berapa persen dari warga wajib edukasi di suatu "
        "desa/kecamatan yang SUDAH benar-benar diedukasi, dihitung dari "
        "akumulasi laporan kegiatan yang diinput Petugas. Rumusnya:\n\n"
        "Jumlah Orang yang Sudah Diedukasi (dari laporan) ÷ Warga Wajib "
        "Diedukasi × 100% (dibatasi maksimal 100%).\n\n"
        "Bedanya dengan Persentase Prioritas: Prioritas itu porsi/target "
        "dari data populasi (statis), Realisasi itu capaian sungguhan di "
        "lapangan (terus berubah tiap ada laporan baru masuk)."
    ),
    "rentan": (
        "**Kelompok Rentan** mencakup balita/lansia, disabilitas, ibu "
        "hamil, dan warga miskin -- kelompok yang butuh perhatian ekstra "
        "saat evakuasi. **Persentase Rentan** dihitung dari:\n\n"
        "Jumlah Kelompok Rentan ÷ Warga Wajib Diedukasi (untuk desa) atau "
        "÷ Warga Terpapar (untuk kecamatan) × 100%\n\n"
        "Dikategorikan: **Tinggi** (≥ 40%), **Sedang** (20% - 39,9%), "
        "**Rendah** (< 20%)."
    ),
    "warna": (
        "**Arti warna di peta:**\n\n"
        "- *Kecamatan* (berdasarkan Persentase Prioritas Kecamatan): 🔴 "
        "Tinggi (≥ 8%), 🟠 Sedang (4% - 7,9%), 🔵 Rendah (< 4%)\n"
        "- *Desa* (berdasarkan Persentase Prioritas Desa): merah tua = "
        "Sangat Tinggi (≥ 30%), merah = Tinggi (15% - 29,9%), oranye = "
        "Sedang (5% - 14,9%), kuning = Rendah (< 5%)\n\n"
        "Ambang batas kecamatan sengaja lebih kecil karena itu hitungan "
        "pangsa dari total SELURUH kecamatan (16 kecamatan), jadi "
        "rata-rata wajar per kecamatan sekitar 6%."
    ),
    "risiko": (
        "**Kelas Risiko** adalah klasifikasi bahaya tanah longsor per "
        "kecamatan yang berasal langsung dari data sumber BPBD (bukan "
        "hasil hitungan aplikasi ini) -- jadi sifatnya informasi "
        "tambahan, terpisah dari Persentase Prioritas maupun Realisasi."
    ),
    "topografi": (
        "**Topografi/Ketinggian Wilayah** menunjukkan apakah suatu "
        "kecamatan berada di dataran rendah (< 500 mdpl), dataran "
        "sedang/perbukitan (500-999 mdpl), atau dataran tinggi/"
        "pegunungan (≥ 1000 mdpl). Data ini dipakai sebagai FAKTOR "
        "TAMBAHAN yang digabung dengan Kelas Risiko asli untuk "
        "menghasilkan **Kelas Risiko + Topografi**, karena wilayah "
        "dataran tinggi umumnya punya lereng lebih curam dan curah "
        "hujan orografis lebih tinggi sehingga potensi longsornya "
        "cenderung lebih besar.\n\n"
        "Rumusnya:\n"
        "Skor dasar (Rendah=1, Sedang=2, Tinggi=3) + bobot topografi "
        "(Dataran Rendah=0, Sedang/Perbukitan=+0.5, Tinggi/Pegunungan="
        "+1) → Kategori Gabungan (≥3 Tinggi, ≥1.5 Sedang, sisanya "
        "Rendah)."
    ),
}


def cari_penjelasan_glosarium(pesan):
    """
    Kalau pesan user kedeteksi sedang menanyakan PENJELASAN istilah/
    rumus (bukan menanyakan data desa/kecamatan tertentu), kembalikan
    jawaban glosarium yang relevan. Kalau kata tanya penjelasan
    terdeteksi tapi topik spesifiknya tidak ketemu, kembalikan
    ringkasan semua istilah sekaligus. Kalau bukan pertanyaan
    penjelasan sama sekali, return None (lanjut ke pencarian
    desa/kecamatan seperti biasa).
    """

    kata_tanya_penjelasan = [
        'apa itu', 'apa arti', 'artinya apa', 'apa maksud', 'maksudnya',
        'jelaskan', 'jelasin', 'rumus', 'cara hitung', 'cara menghitung',
        'definisi', 'bedanya', 'beda antara', 'apa bedanya', 'arti warna',
        'kenapa warna', 'kok warna', 'help', 'bantuan', 'cara pakai',
    ]

    if not any(kt in pesan for kt in kata_tanya_penjelasan):
        return None

    # Kalau pesan menyinggung prioritas & realisasi sekaligus (mis.
    # "beda prioritas dan realisasi"), kasih penjelasan gabungan
    # duluan -- jangan sampai berhenti di salah satunya saja.
    if 'prioritas' in pesan and 'realisasi' in pesan:
        return (
            GLOSARIUM_EDUKASI['prioritas']
            + "\n\n---\n\n"
            + GLOSARIUM_EDUKASI['realisasi']
        )

    if 'realisasi' in pesan:
        return GLOSARIUM_EDUKASI['realisasi']

    if 'prioritas' in pesan:
        return GLOSARIUM_EDUKASI['prioritas']

    if 'rentan' in pesan:
        return GLOSARIUM_EDUKASI['rentan']

    if 'warna' in pesan or 'legend' in pesan or 'legenda' in pesan:
        return GLOSARIUM_EDUKASI['warna']

    if 'topografi' in pesan or 'ketinggian' in pesan or 'dataran' in pesan or 'mdpl' in pesan:
        return GLOSARIUM_EDUKASI['topografi']

    if 'risiko' in pesan:
        return GLOSARIUM_EDUKASI['risiko']

    if 'beda' in pesan:
        # Nanya "beda"/"bedanya" tapi tidak spesifik istilah mana ->
        # langsung kasih perbandingan Prioritas vs Realisasi, karena
        # itu yang paling sering bikin bingung.
        return (
            GLOSARIUM_EDUKASI['prioritas']
            + "\n\n---\n\n"
            + GLOSARIUM_EDUKASI['realisasi']
        )

    # Kata tanya penjelasan terdeteksi tapi topiknya tidak spesifik
    # -> kasih ringkasan semua istilah sekaligus.
    return (
        "Berikut istilah & rumus yang dipakai di sistem ini:\n\n"
        + GLOSARIUM_EDUKASI['prioritas']
        + "\n\n---\n\n"
        + GLOSARIUM_EDUKASI['realisasi']
        + "\n\n---\n\n"
        + GLOSARIUM_EDUKASI['rentan']
        + "\n\n---\n\n"
        + GLOSARIUM_EDUKASI['warna']
        + "\n\n---\n\n"
        + GLOSARIUM_EDUKASI['risiko']
        + "\n\n---\n\n"
        + GLOSARIUM_EDUKASI['topografi']
    )


TIPS_GLOSARIUM_FOOTER = (
    "\n\n💡 *Bingung sama istilah di atas? Ketik misalnya "
    "\"apa itu persentase realisasi\" atau \"beda prioritas dan realisasi\".*"
)


# ========================================================
# ROUTES - CHATBOT SEDERHANA (/api/chat)
# ========================================================
# Catatan: ini chatbot rule-based (bukan AI generatif) yang
# menjawab berdasarkan data desa/kecamatan yang sudah dimuat
# di memori (desa_dict & kec_dict), ditambah glosarium istilah
# di atas. Bisa di-upgrade ke model AI sungguhan nanti kalau
# sudah ada API key-nya.
# ========================================================

SYSTEM_PROMPT = """
Kamu adalah Asisten AI Resmi BPBD Kabupaten Bandung Barat (KBB).

[ATURAN RESPONS]
1. Padat, lugas, dan langsung ke inti (to the point). Hindari kata-kata pembuka/penutup yang bertele-tele.
2. Gunakan format bullet points atau bolding agar informasi mudah dipindai (scannable).
3. Tetap detail dalam menyajikan penjelasan tanpa membuang kata-kata.

[BASIS PENGETAHUAN & RUMUS]
Gunakan acuan rumus dan istilah resmi BPBD KBB berikut jika pengguna bertanya:

1. Persentase Realisasi Edukasi:
   - Rumus: (Jumlah Warga Teredukasi Aktual / Warga Wajib Edukasi) x 100%
   - Penjelasan: Persentase warga yang sudah berhasil diedukasi secara nyata berdasarkan laporan petugas lapangan.

2. Persentase Prioritas Edukasi:
   - Rumus: (Jumlah Teredukasi Desa / Total Warga Terpapar Kecamatan) x 100%
   - Penjelasan: Bobot kontribusi tingkat edukasi suatu desa terhadap keseluruhan warga terpapar di skala kecamatan/kabupaten.

3. Kelompok Rentan:
   - Rumus: (Total Rentan / Total Populasi Wilayah) x 100%
   - Penjelasan: Mencakup lansia, balita, penyandang disabilitas, dan warga berpenyakit kronis.

4. Warga Wajib Edukasi:
   - Penjelasan: Jumlah warga di area rawan bencana yang menjadi target prioritas intervensi edukasi.

Jika pengguna bertanya di luar konteks kebencanaan atau data spesifik, jawablah secara umum, profesional, dan tetap relevan dengan KBB/kesiapsiagaan bencana. Jika pengguna mencari data angka desa/kecamatan tertentu tetapi namanya tidak terdeteksi di database, ingatkan mereka untuk menyebutkan nama desa/kecamatan secara eksplisit.
"""
def cari_wilayah_berdasarkan_realisasi(pesan):
    """
    Mencari desa/kecamatan berdasarkan persentase realisasi edukasi aktual.
    Digunakan untuk pertanyaan seperti:
    - daerah mana yang sudah 100% diedukasi
    - desa mana yang sudah selesai edukasi
    - wilayah mana yang belum 100%
    """

    pesan_lower = pesan.lower()

    # Apakah pertanyaan membahas realisasi edukasi?
    kata_edukasi = [
        "edukasi",
        "teredukasi",
        "edukasi",
        "realisasi",
        "sudah diedukasi",
        "sudah teredukasi"
    ]

    if not any(kata in pesan_lower for kata in kata_edukasi):
        return None

    # -----------------------------------------
    # Tentukan target persentase
    # -----------------------------------------

    target = None

    if "100%" in pesan_lower or "100 persen" in pesan_lower:
        target = 100

    elif "50%" in pesan_lower or "50 persen" in pesan_lower:
        target = 50

    elif "75%" in pesan_lower or "75 persen" in pesan_lower:
        target = 75

    # -----------------------------------------
    # Ambil semua desa
    # -----------------------------------------

    desa_hasil = []

    for d in desa_dict.values():

        persen = safe_number(
            d.get("Persen_Realisasi_Edukasi_Desa", 0)
        )

        if target is not None:

            if persen >= target:
                desa_hasil.append({
                    "wilayah": d.get("Desa", "-"),
                    "kecamatan": d.get("Kecamatan", "-"),
                    "persen": persen,
                    "jumlah": d.get(
                        "Jumlah_Teredukasi_Aktual_Desa", 0
                    ),
                    "target": d.get(
                        "Warga_Wajib_Edukasi_Desa", 0
                    )
                })

    # -----------------------------------------
    # Ambil semua kecamatan
    # -----------------------------------------

    kec_hasil = []

    for k in kec_dict.values():

        persen = safe_number(
            k.get("Persen_Realisasi_Edukasi_Kec", 0)
        )

        if target is not None:

            if persen >= target:
                kec_hasil.append({
                    "wilayah": k.get("Kecamatan", "-"),
                    "persen": persen,
                    "jumlah": k.get(
                        "Jumlah_Teredukasi_Aktual_Kec", 0
                    ),
                    "target": k.get(
                        "Total_Warga_Wajib_Edukasi_Kecamatan", 0
                    )
                })

    # -----------------------------------------
    # Jika tidak ada target angka
    # -----------------------------------------

    if target is None:
        return None

    # -----------------------------------------
    # Buat jawaban
    # -----------------------------------------

    if not desa_hasil and not kec_hasil:
        return (
            f"Belum ada wilayah yang mencapai "
            f"**{target}% realisasi edukasi**."
        )

    reply = (
        f"### Wilayah dengan realisasi edukasi ≥ {target}%\n\n"
    )

    if desa_hasil:

        reply += "**Desa:**\n"

        for d in sorted(
            desa_hasil,
            key=lambda x: x["persen"],
            reverse=True
        ):
            reply += (
                f"- **Desa {d['wilayah']}** "
                f"(Kec. {d['kecamatan']}) — "
                f"**{d['persen']:.2f}%** "
                f"({int(d['jumlah'])} dari "
                f"{int(d['target'])} warga)\n"
            )

    if kec_hasil:

        reply += "\n**Kecamatan:**\n"

        for k in sorted(
            kec_hasil,
            key=lambda x: x["persen"],
            reverse=True
        ):
            reply += (
                f"- **Kecamatan {k['wilayah']}** — "
                f"**{k['persen']:.2f}%** "
                f"({int(k['jumlah'])} dari "
                f"{int(k['target'])} warga)\n"
            )

    return reply


# 3. Route API Chatbot
@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    pesan = (payload.get("message") or "").strip().lower()

    if not pesan:
        return jsonify(
            {
                "reply": "Silakan ketik nama desa atau kecamatan yang ingin Anda tanyakan."
            }
        )

    if any(kata in pesan for kata in ["reset", "kembalikan", "awal"]):
        return jsonify({"reply": "Baik, peta dikembalikan ke tampilan awal."})

    # 1. Cek Glosarium
    penjelasan = cari_penjelasan_glosarium(pesan)
    
    if penjelasan:
        return jsonify({"reply": penjelasan})
    
    
    # 2. Cek pertanyaan prioritas daerah
    hasil_prioritas = cari_prioritas_daerah(pesan)
    
    if hasil_prioritas:
        return jsonify({
            "reply": hasil_prioritas
        })


# 3. Cari kecocokan nama desa
    
    # 2. Cek pertanyaan skor prioritas wilayah
    hasil_skor = cari_skor_prioritas_wilayah(pesan)
    
    if hasil_skor:
        return jsonify({
            "reply": hasil_skor
        })
    
    # 2. Cari wilayah berdasarkan persentase realisasi edukasi
    hasil_realisasi = cari_wilayah_berdasarkan_realisasi(pesan)
    
    if hasil_realisasi:
        return jsonify({
            "reply": hasil_realisasi
        })


# 3. Cari kecocokan nama desa
    # 2. Cari kecocokan nama desa (lebih spesifik)
    desa_cocok = None
    for key, d in desa_dict.items():
        if key in pesan or str(d.get("Desa", "")).lower() in pesan:
            desa_cocok = d
            break

    if desa_cocok:
        reply = (
            f"**Desa {desa_cocok.get('Desa')}** (Kec. {desa_cocok.get('Kecamatan')})\n\n"
            f"- Warga wajib edukasi: **{int(desa_cocok.get('Warga_Wajib_Edukasi_Desa', 0))} orang**\n"
            f"- Kelompok rentan: **{int(desa_cocok.get('Total_Rentan_Desa', 0))} orang "
            f"({desa_cocok.get('Persen_Rentan_Desa', 0)}%)** — kategori *{desa_cocok.get('Kategori_Rentan_Desa', '-')}*\n"
            f"- Persentase prioritas warga yang teredukasi (terhadap warga terpapar kecamatan): "
            f"**{desa_cocok.get('Persen_Teredukasi_Desa', 0)}%**\n"
            f"- Persentase realisasi edukasi (real, dari laporan petugas): "
            f"**{desa_cocok.get('Persen_Realisasi_Edukasi_Desa', 0)}%** "
            f"({int(desa_cocok.get('Jumlah_Teredukasi_Aktual_Desa', 0))} dari "
            f"{int(desa_cocok.get('Warga_Wajib_Edukasi_Desa', 0))} warga wajib edukasi)"
            f"{TIPS_GLOSARIUM_FOOTER}"
        )
        return jsonify({"reply": reply})

    # 3. Cari kecocokan nama kecamatan
    kec_cocok = None
    for key, k in kec_dict.items():
        if key in pesan or str(k.get("Kecamatan", "")).lower() in pesan:
            kec_cocok = k
            break

    if kec_cocok:
        reply = (
            f"**Kecamatan {kec_cocok.get('Kecamatan')}**\n\n"
            f"- Total warga terpapar: **{int(kec_cocok.get('Terpapar_Kecamatan', 0))} orang**\n"
            f"- Kelompok rentan: **{int(kec_cocok.get('Total_Rentan_Kec', 0))} orang "
            f"({kec_cocok.get('Persen_Rentan_Kec', 0)}%)** — kategori *{kec_cocok.get('Kategori_Rentan_Kec', '-')}*\n"
            f"- Kelas risiko (dasar): **{kec_cocok.get('Kelas_Risiko_Kec', '-')}**\n"
            f"- Topografi wilayah: **{kec_cocok.get('Kategori_Topografi_Kec', '-')}** "
            f"(±{int(kec_cocok.get('Elevasi_M_Kec') or 0)} mdpl)\n"
            f"- Kelas risiko + topografi: **{kec_cocok.get('Kategori_Risiko_Gabungan_Kec', '-')}** "
            f"(skor {kec_cocok.get('Skor_Risiko_Gabungan_Kec', 0)})\n"
            f"- Persentase prioritas warga yang teredukasi (dari total seluruh KBB): "
            f"**{kec_cocok.get('Persen_Teredukasi_Kecamatan', 0)}%**\n"
            f"- Persentase realisasi edukasi (real, dari laporan petugas): "
            f"**{kec_cocok.get('Persen_Realisasi_Edukasi_Kec', 0)}%** "
            f"({int(kec_cocok.get('Jumlah_Teredukasi_Aktual_Kec', 0))} dari "
            f"{int(kec_cocok.get('Total_Warga_Wajib_Edukasi_Kecamatan', 0))} warga wajib edukasi)"
            f"{TIPS_GLOSARIUM_FOOTER}"
        )
        return jsonify({"reply": reply})

    # 4. RAG: ambil potongan panduan yang paling relevan sebelum meminta jawaban AI.
    rag_results = retrieve(pesan, top_k=6, min_score=0.08)
    rag_context = build_context(rag_results)

    if groq_client is None:
        if rag_context:
            return jsonify({"reply": "Saya menemukan panduan yang relevan, tetapi layanan AI belum aktif. Admin perlu memastikan GROQ_API_KEY tersedia agar panduan dapat dirangkum menjadi rekomendasi."})
        return jsonify({"reply": "Layanan AI belum aktif dan belum ada panduan yang relevan. Silakan hubungi Admin."})

    system_rag = SYSTEM_PROMPT + """

    ATURAN RAG BPBD KBB:
    - Gunakan konteks panduan di bawah sebagai sumber utama untuk pertanyaan solusi/penanganan.
    - Jangan mengarang aturan, SOP, angka, atau prosedur yang tidak ada di konteks.
    - Jika konteks tidak cukup, katakan dengan jelas bahwa informasi belum ditemukan dan sarankan Admin menambahkan panduan.
    - Bedakan fakta dari dokumen dengan rekomendasi analitis.
    - Untuk tindakan lapangan yang berisiko tinggi, arahkan pengguna mengikuti SOP resmi BPBD/instansi berwenang.
    - Sebutkan nama sumber dokumen secara singkat di bagian akhir jawaban.

    KONTEKS DOKUMEN YANG TERAMBIL:
    """ + (rag_context or "Tidak ada dokumen panduan yang relevan.")

    messages = [
        {"role": "system", "content": system_rag},
        {"role": "user", "content": pesan},
    ]

    available_models = get_available_groq_models()
    if available_models:
        model_candidates = [m for m in dict.fromkeys(GROQ_SAFE_FALLBACKS) if m in available_models]
        if not model_candidates:
            # Pilih model chat generatif yang umum tersedia, bukan model guard/whisper.
            preferred_prefixes = ("llama-", "openai/gpt-oss-", "qwen/", "moonshotai/")
            model_candidates = [m for m in available_models if m.startswith(preferred_prefixes)]
    else:
        model_candidates = list(dict.fromkeys(GROQ_SAFE_FALLBACKS))

    last_error = None
    for model_name in model_candidates:
        try:
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
            )
            ai_reply = completion.choices[0].message.content
            if ai_reply:
                return jsonify({
                    "reply": ai_reply,
                    "sources": [r["source"] for r in rag_results],
                    "model": model_name,
                })
        except Exception as exc:
            last_error = exc
            print(f"[AI/RAG] Model {model_name} gagal: {type(exc).__name__}: {exc}")

    if rag_context:
        return jsonify({
            "reply": (
                "Saya menemukan panduan yang relevan, tetapi layanan AI sedang tidak dapat "
                "menghasilkan jawaban. Berikut sumber panduan yang ditemukan: "
                + ", ".join(dict.fromkeys(r["source"] for r in rag_results))
                + ". Admin dapat memeriksa GROQ_API_KEY/GROQ_MODEL di environment dan daftar model yang diizinkan pada Groq Project. "
                + (f" Detail teknis: {type(last_error).__name__}." if last_error else "")
            ),
            "sources": [r["source"] for r in rag_results],
        }), 503

    return jsonify({"reply": "Panduan yang relevan belum ditemukan dan layanan AI sedang tidak aktif. Silakan Admin menambahkan panduan yang sesuai."}), 503

# ========================================================
# RUN
# ========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )