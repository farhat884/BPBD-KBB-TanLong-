import json
import os
import html
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from functools import wraps
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from groq import Groq

import folium

from ml_engine import get_ml_clustered_data, clean_name, kategori_rentan, update_desa_excel
import json
import os
import re

firebase_json_env = os.getenv(
    "FIREBASE_CREDENTIALS"
)  # Sesuaikan nama variabel kamu

if firebase_json_env:
    # Memotong karakter berlebih di luar kurung kurawal '{ ... }'
    match = re.search(r"\{.*\}", firebase_json_env, re.DOTALL)
    if match:
        cred_dict = json.loads(match.group(0))
    else:
        raise ValueError("Struktur JSON Firebase tidak ditemukan.")

#bpbd_kbb_super_rahasia_@2024_jangan_bocor

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-key-hanya-untuk-lokal')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ========================================================
# KONEKSI KE FIREBASE
# ========================================================
# Arahkan ke file JSON yang baru saja Anda buat
cred_path = os.environ.get('FIREBASE_CREDENTIALS', 'firebase_key.json')

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
        ref_laporan = db.reference('laporan_edukasi')
        laporan_data = ref_laporan.get() or {}
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


# Sinkronkan sekali saat server start supaya laporan yang sudah ada
# di Firebase sebelumnya ikut terhitung sejak awal (bukan mulai 0).
sync_realisasi_edukasi()


# ========================================================
# GEOJSON
# ========================================================
 
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
 
 
            if key_k in kec_dict:
 
                persen = safe_number(
                    kec_dict[key_k].get(
                        "Persen_Teredukasi_Kecamatan",
                        0
                    )
                )
 
                # ---------------------------------------------
                # Ambang batas warna kecamatan
                # ---------------------------------------------
                # Persen_Teredukasi_Kecamatan = pangsa warga
                # terpapar (wajib edukasi) kecamatan ini
                # terhadap TOTAL warga terpapar seluruh
                # kecamatan x 100. Karena nilainya adalah
                # "pangsa dari total", makin banyak kecamatan
                # yang dibandingkan makin kecil rata-ratanya
                # (mis. 16 kecamatan -> rata-rata ~6.25%).
                # Ambang di bawah ini dikalibrasi dari sebaran
                # data saat ini (kira-kira dibagi 3 kelompok
                # rata besar/tinggi/rendah) -- sesuaikan lagi
                # kalau jumlah kecamatan berubah.
                # ---------------------------------------------
 
                if persen >= 8:
 
                    fill_color = "#d7191c"
 
                elif persen >= 4:
 
                    fill_color = "#fdae61"
 
                else:
 
                    fill_color = "#2b83ba"
 
 
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
 
 
                tooltip_html = f"""
                <div style="
                    font-family: Arial, sans-serif;
                    min-width: 210px;
                    padding: 4px;
                ">
 
                    <b style="
                        font-size: 13px;
                        color: #2c3e50;
                    ">
                        KEC. {k['Kecamatan'].upper()}
                    </b>
 
                    <br>
 
                    <hr style="
                        margin: 4px 0;
                        border: 0;
                        border-top: 1px solid #ccc;
                    ">
 
                    Warga Terpapar (Wajib Edukasi):
                    <b>
                        {int(k['Terpapar_Kecamatan']):,}
                    </b>
                    jiwa
 
                    <br>
 
                    <u>Detail Kelompok Rentan:</u>
 
                    <br>
 
                    &nbsp;&nbsp;• Balita/Lansia:
                    <b>{int(k['Rentan_BalitaLansia_Kec']):,}</b>
 
                    <br>
 
                    &nbsp;&nbsp;• Disabilitas:
                    <b>{int(k['Rentan_Disabilitas_Kec']):,}</b>
 
                    <br>
 
                    &nbsp;&nbsp;• Ibu Hamil:
                    <b>{int(k['Rentan_IbuHamil_Kec']):,}</b>
 
                    <br>
 
                    Persentase Rentan:
                    <b>
                        {k['Persen_Rentan_Kec']:.2f}%
                    </b>
                    ({k['Kategori_Rentan_Kec']})
 
                    <hr style="
                        margin: 4px 0;
                        border: 0;
                        border-top: 1px solid #ccc;
                    ">
 
                    Persentase Prioritas Warga
                    yang Teredukasi:
                    <b style="
                        color: #d35400;
                    ">
                        {k['Persen_Teredukasi_Kecamatan']:.2f}%
                    </b>

                    <br>

                    Persentase Realisasi Edukasi (Real):
                    <b style="
                        color: #16a34a;
                    ">
                        {k.get('Persen_Realisasi_Edukasi_Kec', 0):.2f}%
                    </b>
                    <br>
                    <span style="font-size: 10px;">
                        ({int(k.get('Jumlah_Teredukasi_Aktual_Kec', 0)):,}
                        dari {int(k.get('Total_Warga_Wajib_Edukasi_Kecamatan', 0)):,}
                        warga wajib edukasi)
                    </span>
 
                    <hr style="
                        margin: 4px 0;
                        border: 0;
                        border-top: 1px solid #ccc;
                    ">
 
                    <i style="
                        font-size: 10px;
                        color: #e67e22;
                    ">
                        👉 Klik wilayah untuk
                        melihat desa
                    </i>
 
                </div>
                """
 
            else:
 
                tooltip_html = (
                    f"<b>Kecamatan: "
                    f"{raw_kec}</b>"
                )
 
 
            folium.GeoJson(
 
                feature,
 
                style_function=
                    style_kecamatan,
 
                highlight_function=lambda x: {
                    "weight": 3.5,
                    "color": "#000000",
                    "fillOpacity": 0.95
                },
 
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
 
 
    def style_desa(feature):
 
        raw_desa = get_nama_desa(
            feature["properties"]
        )
 
        key_d = clean_name(
            raw_desa
        )
 
 
        # DEFAULT
        fill_color = "#cccccc"
 
 
        if key_d in desa_dict:
 
            persen = safe_number(
                desa_dict[key_d].get(
                    "Persen_Teredukasi_Desa",
                    0
                )
            )
 
 
            fill_color = get_color_desa(
                persen
            )
 
 
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
                f"desa-parent-{raw_kec_parent}"
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
 
 
            tooltip_html = f"""
            <b>
                DESA {str(
                    d.get('Desa', '')
                ).upper()}
            </b>
 
            <br>
 
            Wajib Diedukasi:
            {int(
                d.get(
                    'Warga_Wajib_Edukasi_Desa',
                    0
                )
            ):,} jiwa
 
            <br>
 
            Rentan: {persen_rentan_desa:.2f}%
            ({kategori_rentan_desa})
 
            <br>
 
            Prioritas Warga Teredukasi:
            <b>
                {persen_teredukasi:.2f}%
            </b>

            <br>

            Realisasi Edukasi (Real):
            <b style="color: #16a34a;">
                {d.get('Persen_Realisasi_Edukasi_Desa', 0):.2f}%
            </b>
            """
 
 
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
 
            highlight_function=lambda x: {
                "weight": 2.5,
                "color": "#000000",
                "fillOpacity": 0.95
            },
 
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

        <b style="
            font-size:13px;
            color:#00ffcc;
        ">
            Persentase Prioritas Warga
            yang Teredukasi (Kecamatan)
        </b>

        <br>

        <span style="
            font-size:10px;
            color:#cbd5e1;
        ">
            % Terpapar Kecamatan ini /
            Total Terpapar Seluruh Kecamatan
        </span>

        <hr style="
            margin:6px 0;
            border:0;
            border-top:
                1px solid
                rgba(0,255,204,0.3);
        ">


        <div style="
            margin-bottom:4px;
        ">

            <i style="
                background:#d7191c;
                width:12px;
                height:12px;
                display:inline-block;
                vertical-align:middle;
                margin-right:8px;
                border-radius:2px;
            "></i>

            <b>Tinggi</b>
            (≥ 8%)

        </div>


        <div style="
            margin-bottom:4px;
        ">

            <i style="
                background:#fdae61;
                width:12px;
                height:12px;
                display:inline-block;
                vertical-align:middle;
                margin-right:8px;
                border-radius:2px;
            "></i>

            <b>Sedang</b>
            (4% - 7.9%)

        </div>


        <div>

            <i style="
                background:#2b83ba;
                width:12px;
                height:12px;
                display:inline-block;
                vertical-align:middle;
                margin-right:8px;
                border-radius:2px;
            "></i>

            <b>Rendah</b>
            (&lt; 4%)

        </div>

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

        <b style="
            font-size:13px;
            color:#00ffcc;
        ">
            Persentase Prioritas Warga
            yang Teredukasi (Desa)
        </b>

        <br>

        <span style="
            font-size:10px;
            color:#cbd5e1;
        ">
            % Wajib Diedukasi Desa /
            Terpapar Kecamatan
        </span>

        <hr style="
            margin:6px 0;
            border:0;
            border-top:
                1px solid
                rgba(0,255,204,0.3);
        ">


        <div style="
            margin-bottom:4px;
        ">

            <i style="
                background:#800026;
                width:12px;
                height:12px;
                display:inline-block;
                vertical-align:middle;
                margin-right:8px;
                border-radius:2px;
            "></i>

            <b>Sangat Tinggi</b>
            (≥ 30%)

        </div>


        <div style="
            margin-bottom:4px;
        ">

            <i style="
                background:#BD0026;
                width:12px;
                height:12px;
                display:inline-block;
                vertical-align:middle;
                margin-right:8px;
                border-radius:2px;
            "></i>

            <b>Tinggi</b>
            (15% - 29.9%)

        </div>


        <div style="
            margin-bottom:4px;
        ">

            <i style="
                background:#FD8D3C;
                width:12px;
                height:12px;
                display:inline-block;
                vertical-align:middle;
                margin-right:8px;
                border-radius:2px;
            "></i>

            <b>Sedang</b>
            (5% - 14.9%)

        </div>


        <div>

            <i style="
                background:#FED976;
                width:12px;
                height:12px;
                display:inline-block;
                vertical-align:middle;
                margin-right:8px;
                border-radius:2px;
            "></i>

            <b>Rendah</b>
            (&lt; 5%)

        </div>

    </div>

    </div>
    <!-- /#panel-kiri-bawah -->

    """


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
                    /kec\.?/g,
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
# ROUTES - HALAMAN PUBLIK
# ========================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/kondisi")
def kondisi():

    peta_html = generate_map()

    return render_template(
        "kondisi.html",
        peta_html=peta_html,
        data_dict=desa_dict,
        kec_dict=kec_dict
    )


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


# ========================================================
# ROUTES - DASHBOARD ADMIN
# ========================================================

@app.route("/dashboard-admin")
@role_required('admin')
def dashboard_admin():

    # --- Daftar akun (users) ---
    ref_users = db.reference('users')
    users_data = ref_users.get() or {}

    # --- Daftar laporan edukasi dari semua petugas ---
    ref_laporan = db.reference('laporan_edukasi')
    laporan_data = ref_laporan.get() or {}
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

    return render_template(
        "dashboard_admin.html",
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

    total_rentan = bl + miskin + disabilitas
    persen_rentan = round((total_rentan / warga * 100), 2) if warga > 0 else 0.0

    data = desa_dict[desa_key]
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

    # Simpan juga ke file Excel sumber supaya perubahan tidak hilang saat server di-restart
    update_desa_excel(
        app.root_path,
        data.get('Desa', ''),
        data.get('Kecamatan', ''),
        warga, bl, miskin, disabilitas
    )

    flash(f"Data desa {data.get('Desa', '-')} berhasil diperbarui.")
    return redirect(url_for('dashboard_admin'))


# ========================================================
# ROUTES - DASHBOARD PETUGAS
# ========================================================

@app.route("/dashboard-petugas")
@role_required('petugas', 'admin')
def dashboard_petugas():

    ref_laporan = db.reference('laporan_edukasi')
    laporan_data = ref_laporan.get() or {}
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
        "dashboard_petugas.html",
        laporan_list=laporan_list,
        total_orang_teredukasi_saya=total_orang_teredukasi_saya,
        desa_by_kecamatan=desa_by_kecamatan,
    )


@app.route("/petugas/lapor", methods=['POST'])
@role_required('petugas', 'admin')
def petugas_lapor():
    kecamatan = request.form.get('kecamatan', '').strip()
    desa = request.form.get('desa', '').strip()
    deskripsi = request.form.get('deskripsi', '').strip()
    jumlah_kk = int(safe_number(request.form.get('jumlah_kk', 0)))

    if not kecamatan or not desa or not deskripsi or jumlah_kk <= 0:
        flash("Kecamatan, desa, jumlah KK yang diedukasi, dan deskripsi kegiatan wajib diisi dengan benar.")
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
            f"- Kelas risiko: **{kec_cocok.get('Kelas_Risiko_Kec', '-')}**\n"
            f"- Persentase prioritas warga yang teredukasi (dari total seluruh KBB): "
            f"**{kec_cocok.get('Persen_Teredukasi_Kecamatan', 0)}%**\n"
            f"- Persentase realisasi edukasi (real, dari laporan petugas): "
            f"**{kec_cocok.get('Persen_Realisasi_Edukasi_Kec', 0)}%** "
            f"({int(kec_cocok.get('Jumlah_Teredukasi_Aktual_Kec', 0))} dari "
            f"{int(kec_cocok.get('Total_Warga_Wajib_Edukasi_Kecamatan', 0))} warga wajib edukasi)"
            f"{TIPS_GLOSARIUM_FOOTER}"
        )
        return jsonify({"reply": reply})

    # 4. GROQ AI FALLBACK (Dipanggil jika tidak ada pencarian lokal yang cocok)
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pesan},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        ai_reply = completion.choices[0].message.content
        return jsonify({"reply": ai_reply})

    except Exception as e:
        return jsonify(
            {
                "reply": (
                    "Maaf, saya belum menemukan data spesifik untuk itu. Coba ketik nama **desa** atau **kecamatan** "
                    'di wilayah Kabupaten Bandung Barat, misalnya "kondisi desa Cikalonglor" atau "kecamatan Lembang".'
                )
            }
        )

# ========================================================
# RUN
# ========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )
