import os
import pandas as pd

import re

from elevation_engine import hitung_elevasi_kecamatan, skor_topografi

def clean_name(s):
    if pd.isna(s):
        return ''

    s = str(s).strip().lower()

    # Hilangkan kata kecamatan/kec.
    s = re.sub(r'\bkecamatan\b', '', s)
    s = re.sub(r'\bkec\.?\b', '', s)

    # Hilangkan underscore, spasi, tanda baca
    s = re.sub(r'[^a-z0-9]', '', s)

    return s

def bagikan_kuota_sisa_terbesar(total, bobot):
    """
    Membagi satu angka `total` (integer, mis. jumlah KK peserta acara)
    ke beberapa "wadah" (mis. desa) berdasarkan `bobot` masing-masing,
    memakai METODE KUOTA + SISA TERBESAR (Largest Remainder Method /
    Hamilton Method).

    DIPAKAI UNTUK KASUS: petugas mengisi laporan edukasi tapi peserta
    acara berasal dari SELURUH Kabupaten Bandung Barat (bukan dari satu
    desa tertentu), jadi tidak ada kepastian asal per desa. Daripada
    dibiarkan tidak tercatat sama sekali atau ditumpuk di satu desa
    secara sembarangan, jumlah peserta dialokasikan ke semua desa
    memakai asumsi yang eksplisit & bisa diaudit -- BUKAN cuma dibagi
    rata secara serampangan.

    DASAR TEORI / KENAPA METODE INI:
    - Kalau `bobot` semua desa dibuat SAMA (mis. semua = 1), ini
      menerapkan "principle of indifference" (asas tak berbeda /
      Laplace): tanpa informasi tambahan, tiap desa dianggap punya
      peluang sama menyumbang peserta -- estimasi paling netral saat
      benar-benar tidak ada data asal peserta.
    - Kalau `bobot` diisi proporsi populasi (mis. Warga_Wajib_Edukasi_Desa),
      ini menjadi estimasi post-stratifikasi (proporsional terhadap
      ukuran populasi tiap desa) -- opsi ini TIDAK dipakai untuk
      request saat ini (user memilih asumsi rata/indifference), tapi
      fungsi ini dibuat generik supaya bisa dipakai untuk keduanya.
    - Kenapa bukan cuma round() / pembagian rata biasa: pembulatan
      independen per desa bisa membuat TOTAL akhir meleset dari jumlah
      KK yang sebenarnya dilaporkan petugas (mis. 200 KK dibagi 16
      kecamatan = 12.5 per kecamatan -> kalau asal dibulatkan, totalnya
      bisa jadi 192 atau 208, bukan 200 lagi). Metode Kuota + Sisa
      Terbesar menjamin jumlah alokasi akhir PERSIS SAMA dengan total
      input, dengan cara:
        1. Hitung kuota eksak tiap wadah = (bobot wadah / total bobot) x total.
        2. Alokasikan dulu bagian bulat ke bawah (floor) dari tiap kuota.
        3. Sisa unit yang belum terbagi (karena pembulatan ke bawah)
           diberikan satu-satu ke wadah dengan SISA DESIMAL terbesar,
           sampai sisa habis.
      Metode ini historisnya dipakai untuk alokasi kursi DPR/DPRD di
      Pemilu Indonesia 1999 & 2004 (dikenal sebagai "kuota Hare"), dan
      di AS dikenal sebagai Hamilton's Method untuk pembagian kursi
      House of Representatives (dipakai 1852-1900).

    Parameter
    ---------
    total : int
        Jumlah unit yang mau dibagi (mis. total KK peserta acara).
    bobot : dict {key: angka_bobot_non_negatif}
        Bobot tiap wadah. Kalau mau pembagian RATA (asas tak berbeda),
        isi semua value dengan angka yang sama, mis. 1.

    Return
    ------
    dict {key: alokasi_int} -- totalnya dijamin persis sama dengan
    `total` (selama total >= 0 dan ada minimal satu bobot > 0).
    """
    try:
        total = int(round(float(total)))
    except (TypeError, ValueError):
        total = 0

    bobot_bersih = {k: max(0.0, float(v or 0)) for k, v in bobot.items()}
    total_bobot = sum(bobot_bersih.values())

    alokasi = {k: 0 for k in bobot_bersih}

    if total <= 0 or total_bobot <= 0:
        return alokasi

    kuota_eksak = {k: (v / total_bobot) * total for k, v in bobot_bersih.items()}

    # Langkah 1: alokasi awal = floor(kuota eksak)
    for k, q in kuota_eksak.items():
        alokasi[k] = int(q)

    # Langkah 2: sisa unit yang belum terbagi karena pembulatan ke bawah
    sisa_unit = total - sum(alokasi.values())

    # Langkah 3: urutkan berdasarkan sisa desimal terbesar, bagikan satu-satu
    urutan_sisa_desimal = sorted(
        bobot_bersih.keys(),
        key=lambda k: kuota_eksak[k] - alokasi[k],
        reverse=True,
    )
    for k in urutan_sisa_desimal[:sisa_unit]:
        alokasi[k] += 1

    return alokasi


def clean_number(val):
    if pd.isna(val) or val == '': 
        return 0
    
    # 1. Ubah ke teks/string dulu
    val_str = str(val).strip()
    
    # 2. Atasi efek float bawaan Pandas (misal "237.0" jadi "237")
    if val_str.endswith('.0'):
        val_str = val_str[:-2]
        
    # 3. Hapus semua sisa tanda baca pemisah ribuan (titik, koma, dan spasi)
    val_str = val_str.replace(' ', '').replace(',', '').replace('.', '')
    
    try:
        return int(val_str)
    except:
        return 0


# ================================================================
# KATEGORI KELOMPOK RENTAN
# ================================================================
# Ambang batas ini masih perkiraan awal, silakan disesuaikan
# lagi dengan standar/acuan resmi yang dipakai di proyekmu.
#   >= 40%           : Tinggi
#   20% - 39.9%      : Sedang
#   < 20%            : Rendah
# ================================================================
def kategori_rentan(persen):
    if persen >= 40:
        return 'Tinggi'
    elif persen >= 20:
        return 'Sedang'
    else:
        return 'Rendah'


# ================================================================
# SKOR RISIKO GABUNGAN (Kelas Risiko asli + Faktor Topografi)
# ================================================================
# Ide dasarnya: "Kelas_Risiko_Kec" yang sudah ada di data Excel
# dijadikan skor dasar, lalu ditambah bobot topografi (dataran
# tinggi/pegunungan menambah potensi, dataran rendah tidak
# menambah). Hasil penjumlahannya diklasifikasi ulang jadi
# kategori risiko baru yang sudah memperhitungkan ketinggian
# wilayah.
# ================================================================
SKOR_DASAR_RISIKO = {
    'rendah': 1,
    'sedang': 2,
    'tinggi': 3,
}


def skor_dasar_dari_kelas_risiko(kelas_risiko):
    key = str(kelas_risiko).strip().lower()
    # Kalau kelas risiko tidak diketahui / belum ada datanya,
    # anggap "sedang" dulu (tidak menganggap remeh, tidak juga
    # langsung dianggap tinggi tanpa dasar).
    return SKOR_DASAR_RISIKO.get(key, 2)


def kategori_risiko_gabungan(skor_gabungan):
    if skor_gabungan >= 3:
        return 'Tinggi'
    elif skor_gabungan >= 1.5:
        return 'Sedang'
    else:
        return 'Rendah'


def terapkan_topografi_manual(target_dict, key, breakdown, level="kecamatan"):
    """
    Menerapkan hasil klasifikasi topografi manual (upload gambar + AI
    vision, lihat topografi_manual.py) ke SATU entri di kec_dict atau
    desa_dict, lalu menghitung ulang Skor_Risiko_Gabungan_Kec &
    Kategori_Risiko_Gabungan_Kec HANYA untuk entri itu.

    Sengaja per-entri (bukan disebar rata ke satu kecamatan/desa),
    supaya topografi TIDAK diasumsikan seragam dalam satu wilayah
    administratif -- ini yang jadi keluhan utama soal pendekatan lama
    (1 kecamatan = 1 elevasi rata-rata).

    Parameter
    ---------
    target_dict : dict
        kec_dict ATAU desa_dict (dict {key: data_wilayah}).
    key : str
        Key wilayah di target_dict (kecamatan_clean, atau desa_key
        berformat "{kecamatan_clean}__{desa_clean}").
    breakdown : dict
        {"persen_rendah": .., "persen_sedang": .., "persen_tinggi": ..}
    level : str
        "kecamatan" | "desa" -- hanya penanda sumber untuk ditampilkan
        di panel admin, tidak memengaruhi rumus skor.

    Return True kalau `key` ditemukan & berhasil diterapkan.
    """
    from topografi_manual import kategori_dominan_dari_breakdown, skor_dari_breakdown

    if key not in target_dict:
        return False

    data = target_dict[key]

    pr = breakdown.get('persen_rendah', 0)
    ps = breakdown.get('persen_sedang', 0)
    pt = breakdown.get('persen_tinggi', 0)

    kategori_dominan = kategori_dominan_dari_breakdown(pr, ps, pt)
    bobot_topo = skor_dari_breakdown(pr, ps, pt)

    data['Elevasi_M_Kec'] = None
    data['Kategori_Topografi_Kec'] = f"{kategori_dominan} (Input Manual)"
    data['Topografi_Breakdown_Kec'] = {
        'persen_rendah': pr,
        'persen_sedang': ps,
        'persen_tinggi': pt,
    }
    data['Topografi_Sumber_Kec'] = 'manual_ai_gambar'
    data['Topografi_Level_Kec'] = level

    skor_dasar = skor_dasar_dari_kelas_risiko(data.get('Kelas_Risiko_Kec', ''))
    data['Skor_Risiko_Gabungan_Kec'] = round(skor_dasar + bobot_topo, 2)
    data['Kategori_Risiko_Gabungan_Kec'] = kategori_risiko_gabungan(
        data['Skor_Risiko_Gabungan_Kec']
    )

    return True


def update_desa_excel(app_root_path, desa_name, kecamatan_name,
                       jumlah_penduduk, umur_rentan, miskin, disabilitas):
    """
    Memperbarui satu baris data desa di file Excel sumber
    (data/Data_Desa_Longsor.xlsx) berdasarkan kecocokan nama desa
    & kecamatan (dibandingkan setelah dibersihkan pakai clean_name).

    Dipanggil dari panel "Kelola Data Desa" di dashboard admin, supaya
    perubahan yang diinput admin tidak hilang saat server di-restart
    (karena data desa awalnya selalu dibaca ulang dari file Excel ini).

    Mengembalikan True jika desa ditemukan & berhasil disimpan,
    False jika desa tidak ditemukan di file Excel.
    """
    excel_desa = os.path.join(app_root_path, 'data', 'Data_Desa_Longsor.xlsx')

    if not os.path.exists(excel_desa):
        return False

    df = pd.read_excel(excel_desa)
    df.columns = df.columns.str.strip().str.lower()

    target_desa = clean_name(desa_name)
    target_kec = clean_name(kecamatan_name)

    baris_ditemukan = False

    for idx, row in df.iterrows():
        if (
            clean_name(row.get('desa', '')) == target_desa
            and clean_name(row.get('kecamatan', '')) == target_kec
        ):
            df.at[idx, 'jumlah_penduduk'] = jumlah_penduduk
            df.at[idx, 'umur_rentan'] = umur_rentan
            df.at[idx, 'miskin'] = miskin
            df.at[idx, 'disabilitas'] = disabilitas
            baris_ditemukan = True
            break

    if baris_ditemukan:
        df.to_excel(excel_desa, index=False)

    return baris_ditemukan


def get_ml_clustered_data(app_root_path):
    # ============================================================
    # 1. BACA DATA DESA
    # ============================================================
    #
    # CATATAN PENTING:
    # Kolom 'jumlah_penduduk' pada file sumber desa BUKAN total
    # penduduk desa, melainkan jumlah WARGA YANG WAJIB DIEDUKASI
    # di desa tersebut. Data total penduduk desa yang sebenarnya
    # memang tidak tersedia di sumber data ini -- tidak masalah,
    # karena yang dipakai untuk seluruh perhitungan memang angka
    # wajib edukasi ini.
    # ============================================================

    excel_desa = os.path.join(app_root_path, 'data', 'Data_Desa_Longsor.xlsx')
    df_desa = pd.DataFrame()
    
    if os.path.exists(excel_desa):
        df_raw_desa = pd.read_excel(excel_desa)
        df_raw_desa.columns = df_raw_desa.columns.str.strip().str.lower()

        records_desa = []
        for _, row in df_raw_desa.iterrows():
            desa_name = str(row.get('desa', '')).strip()
            kec_name = str(row.get('kecamatan', '')).strip()
            if not desa_name or desa_name == 'nan': continue
            
            bl = clean_number(row.get('umur_rentan', 0))
            miskin = clean_number(row.get('miskin', 0))
            disabilitas = clean_number(row.get('disabilitas', 0))

            # 'jumlah_penduduk' = warga wajib diedukasi di desa ini
            warga_wajib_edukasi = clean_number(row.get('jumlah_penduduk', 0))

            total_rentan = bl + miskin + disabilitas

            # % kelompok rentan terhadap warga wajib diedukasi di desa
            persen_rentan = round(
                (total_rentan / warga_wajib_edukasi * 100), 2
            ) if warga_wajib_edukasi > 0 else 0.0

            records_desa.append({
                'Kecamatan': kec_name.title(),
                'Kecamatan_Clean': clean_name(kec_name),
                'Desa': desa_name.title(),
                'Warga_Wajib_Edukasi_Desa': warga_wajib_edukasi,
                'Rentan_BalitaLansia_Desa': bl,
                'Rentan_Miskin_Desa': miskin,
                'Rentan_Disabilitas_Desa': disabilitas,
                'Total_Rentan_Desa': total_rentan,
                'Persen_Rentan_Desa': persen_rentan,
                'Kategori_Rentan_Desa': kategori_rentan(persen_rentan),
            })
        df_desa = pd.DataFrame(records_desa)

    # ============================================================
    # 2. BACA DATA KECAMATAN (Terpapar = warga wajib diedukasi,
    #    tapi dalam skala satu kecamatan)
    # ============================================================

    excel_kec = os.path.join(app_root_path, 'data', 'Data_Potensi_Penduduk_Terpapar_Tanah_Longsor.xlsx')
    df_kec = pd.DataFrame()
    
    if os.path.exists(excel_kec):
        df_raw_kec = pd.read_excel(excel_kec)
        df_raw_kec.columns = df_raw_kec.columns.str.strip().str.lower()
        
        records_kec = []
        for _, row in df_raw_kec.iterrows():
            kec_name = str(row.get('kecamatan', '')).strip()
            if not kec_name or kec_name == 'nan': continue

            # ----------------------------------------------------
            # BUG LAMA ADA DI SINI:
            # kode sebelumnya memanggil row.get('Terpapar', ...)
            # dengan huruf besar, padahal semua nama kolom di atas
            # sudah di-lowercase -> key 'Terpapar' tidak pernah
            # ketemu, nilainya selalu jatuh ke default 0.
            # Diperbaiki jadi 'terpapar' (huruf kecil, sesuai kolom
            # asli di file Excel-mu).
            # ----------------------------------------------------
            terpapar = clean_number(row.get('terpapar', 0))

            rentan_bl = clean_number(row.get('rentan_balita_lansia', 0))
            rentan_disabilitas = clean_number(row.get('rentan_disabilitas', 0))
            rentan_ibu_hamil = clean_number(row.get('rentan_ibu_hamil', 0))
            total_rentan_kec = rentan_bl + rentan_disabilitas + rentan_ibu_hamil

            persen_rentan_kec = round(
                (total_rentan_kec / terpapar * 100), 2
            ) if terpapar > 0 else 0.0

            records_kec.append({
                'Kecamatan_Clean': clean_name(kec_name),
                'Terpapar_Kecamatan': terpapar,
                'Rentan_BalitaLansia_Kec': rentan_bl,
                'Rentan_Disabilitas_Kec': rentan_disabilitas,
                'Rentan_IbuHamil_Kec': rentan_ibu_hamil,
                'Total_Rentan_Kec': total_rentan_kec,
                'Persen_Rentan_Kec': persen_rentan_kec,
                'Kategori_Rentan_Kec': kategori_rentan(persen_rentan_kec),
                'Kelas_Risiko_Kec': str(row.get('kelas_risiko', '')).strip().title(),
            })
        df_kec = pd.DataFrame(records_kec)

    # ============================================================
    # 3. GABUNGKAN & HITUNG PERSENTASE TEREDUKASI
    # ============================================================
    if not df_desa.empty and not df_kec.empty:

        print("=== Kecamatan pada Data Desa ===")
        print(df_desa['Kecamatan_Clean'].unique())

        print("=== Kecamatan pada Data Terpapar Kecamatan ===")
        print(df_kec['Kecamatan_Clean'].unique())

        # Cek kecamatan yang tidak memiliki pasangan
        kec_desa = set(df_desa['Kecamatan_Clean'])
        kec_kec = set(df_kec['Kecamatan_Clean'])

        print("=== Tidak ditemukan di Data Kecamatan ===")
        print(kec_desa - kec_kec)

        print("=== Tidak ditemukan di Data Desa ===")
        print(kec_kec - kec_desa)

        # --------------------------------------------------------
        # % Teredukasi KECAMATAN
        # = Terpapar kecamatan ini / total Terpapar SELURUH
        #   kecamatan x 100
        # Dihitung dari df_kec (satu baris per kecamatan) supaya
        # totalnya tidak double-count saat nanti di-merge ke tiap
        # baris desa.
        # --------------------------------------------------------
        total_terpapar_semua_kec = df_kec['Terpapar_Kecamatan'].sum()

        df_kec['Persen_Teredukasi_Kecamatan'] = df_kec['Terpapar_Kecamatan'].apply(
            lambda t: round((t / total_terpapar_semua_kec * 100), 2)
            if total_terpapar_semua_kec > 0 else 0.0
        )

        # Gabungkan data desa + data kecamatan
        df_final = df_desa.merge(
            df_kec,
            on='Kecamatan_Clean',
            how='left'
        )

        # Kecamatan yang tidak ada pasangannya diisi 0 / default
        # supaya tidak NaN di tampilan
        kolom_numerik_default_0 = [
            'Terpapar_Kecamatan', 'Rentan_BalitaLansia_Kec',
            'Rentan_Disabilitas_Kec', 'Rentan_IbuHamil_Kec',
            'Total_Rentan_Kec', 'Persen_Rentan_Kec',
            'Persen_Teredukasi_Kecamatan'
        ]
        for kolom in kolom_numerik_default_0:
            df_final[kolom] = df_final[kolom].fillna(0)

        df_final['Kategori_Rentan_Kec'] = df_final['Kategori_Rentan_Kec'].fillna('Tidak Ada Data')
        df_final['Kelas_Risiko_Kec'] = df_final['Kelas_Risiko_Kec'].fillna('-')

        # Total warga wajib edukasi per kecamatan (agregasi dari desa)
        kec_totals = (
            df_final
            .groupby('Kecamatan_Clean')['Warga_Wajib_Edukasi_Desa']
            .sum()
            .reset_index()
        )

        kec_totals.rename(
            columns={
                'Warga_Wajib_Edukasi_Desa': 'Total_Warga_Wajib_Edukasi_Kecamatan'
            },
            inplace=True
        )

        df_final = df_final.merge(
            kec_totals,
            on='Kecamatan_Clean',
            how='left'
        )

        # --------------------------------------------------------
        # % Teredukasi DESA
        # = warga wajib diedukasi di desa ini / warga Terpapar
        #   di kecamatannya x 100
        # --------------------------------------------------------
        df_final['Persen_Teredukasi_Desa'] = df_final.apply(
            lambda r: round(
                (
                    r['Warga_Wajib_Edukasi_Desa']
                    / r['Terpapar_Kecamatan']
                    * 100
                ),
                2
            ) if r['Terpapar_Kecamatan'] > 0 else 0.0,
            axis=1
        )

        # ========================================================
        # 4. FAKTOR TOPOGRAFI (dataran tinggi / rendah) + SKOR
        #    RISIKO GABUNGAN
        # ========================================================
        df_final = tambahkan_faktor_topografi(df_final, app_root_path)

        return df_final

    return df_desa


def tambahkan_faktor_topografi(df_final, app_root_path):
    """
    Menambahkan kolom ketinggian & kategori topografi per kecamatan,
    lalu menghitung "Skor_Risiko_Gabungan_Kec" &
    "Kategori_Risiko_Gabungan_Kec" -- yaitu Kelas_Risiko_Kec asli
    yang sudah digabung dengan faktor ketinggian wilayah.

    Kalau data elevasi gagal diambil sama sekali (mis. tidak ada
    file geojson kecamatan), kolom-kolom ini tetap dibuat dengan
    nilai default supaya tidak error di app.py / template.
    """
    try:
        peta_elevasi = hitung_elevasi_kecamatan(app_root_path)
    except Exception as e:
        print(f"⚠️  Gagal menghitung faktor topografi: {e}")
        peta_elevasi = {}

    def _elevasi(row):
        info = peta_elevasi.get(row['Kecamatan_Clean'])
        return info['elevasi_m'] if info else None

    def _kategori_topo(row):
        info = peta_elevasi.get(row['Kecamatan_Clean'])
        return info['kategori_topografi'] if info else 'Tidak Ada Data'

    df_final['Elevasi_M_Kec'] = df_final.apply(_elevasi, axis=1)
    df_final['Kategori_Topografi_Kec'] = df_final.apply(_kategori_topo, axis=1)

    def _skor_gabungan(row):
        skor_dasar = skor_dasar_dari_kelas_risiko(row.get('Kelas_Risiko_Kec', ''))
        bobot_topo = skor_topografi(row['Kategori_Topografi_Kec'])
        return round(skor_dasar + bobot_topo, 2)

    df_final['Skor_Risiko_Gabungan_Kec'] = df_final.apply(_skor_gabungan, axis=1)
    df_final['Kategori_Risiko_Gabungan_Kec'] = df_final['Skor_Risiko_Gabungan_Kec'].apply(
        kategori_risiko_gabungan
    )

    return df_final