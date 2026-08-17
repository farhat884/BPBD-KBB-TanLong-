import os
import pandas as pd

import re

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

        return df_final

    return df_desa