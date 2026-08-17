import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1. DATA JUMLAH WARGA (Dari app.py sebelumnya)
# =========================================================
TOTAL_WARGA_DICT = {
    'saguling': 37987, 'gununghalu': 84529, 'sindangkerta': 80814,
    'rongga': 65316, 'cipongkor': 109952, 'cililin': 104811,
    'cihampelas': 149178, 'batujajar': 118155, 'padalarang': 194806,
    'cipatat': 153339, 'ngamprah': 185000, 'cipeundeuy': 94127,
    'cikalongwetan': 137226, 'cisarua': 85458, 'parongpong': 118593,
    'lembang': 211159,
}

def clean_name(s):
    if not s: return ''
    return str(s).lower().replace('_', '').replace(' ', '').replace('kecamatan', '').replace('kec.', '').strip()

# =========================================================
# 2. BACA EXCEL & LAKUKAN FEATURE ENGINEERING
# =========================================================
# Path ke file Excel kamu (Asumsi folder 'data' ada di folder BPBD yang sama)
base_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(base_dir, 'data', 'Data_Potensi_Penduduk_Terpapar_Tanah_Longsor.xlsx')

if not os.path.exists(excel_path):
    print(f"❌ GAGAL: File Excel tidak ditemukan di:\n{excel_path}")
    exit()

df_raw = pd.read_excel(excel_path)
df_raw.columns = df_raw.columns.str.strip().str.lower()

records = []
for _, row in df_raw.iterrows():
    kec_name = str(row.get('kecamatan', '')).strip()
    if not kec_name or kec_name == 'nan':
        continue
        
    key_clean = clean_name(kec_name)
    total_warga = TOTAL_WARGA_DICT.get(key_clean, 0)
    
    if total_warga == 0:
        continue # Lewati jika total warga 0 agar tidak error pembagian

    teredukasi = int(row.get('teredukasi', 0))
    rentan_bl = int(row.get('rentan_balita_lansia', 0))
    rentan_dis = int(row.get('rentan_disabilitas', 0))
    rentan_bumil = int(row.get('rentan_ibu_hamil', 0))

    # --- FEATURE ENGINEERING ---
    # Mengubah angka mentah menjadi persentase/rasio terhadap total populasi kecamatan
    records.append({
        'Kecamatan': kec_name.capitalize(),
        'Persen_Edukasi': (teredukasi / total_warga) * 100,
        'Rasio_BL': (rentan_bl / total_warga) * 100,
        'Rasio_Dis': (rentan_dis / total_warga) * 100,
        'Rasio_Bumil': (rentan_bumil / total_warga) * 100
    })

# Nah, ini dia variabel 'df' yang dicari oleh Python!
df = pd.DataFrame(records)
print(f"✅ Berhasil memproses data {len(df)} kecamatan.")


# =========================================================
# 3. PROSES MACHINE LEARNING (K-MEANS CLUSTERING)
# =========================================================
# Memilih kolom (features) yang akan dipelajari oleh model
features = ['Rasio_BL', 'Rasio_Dis', 'Rasio_Bumil', 'Persen_Edukasi']
X = df[features]

# Standarisasi data (wajib untuk K-Means agar skalanya seimbang)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Melatih Model K-Means (Kita minta AI membaginya ke 3 kelompok)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X_scaled)

# Menyimpan hasil prediksi ke dalam kolom baru bernama 'Cluster'
df['Cluster'] = kmeans.labels_


# =========================================================
# 4. TAMPILKAN HASILNYA
# =========================================================
print("\n=== HASIL CLUSTERING MACHINE LEARNING ===")
# Mengurutkan data berdasarkan Cluster agar mudah dibaca
df_sorted = df.sort_values(by='Cluster')
print(df_sorted[['Kecamatan', 'Cluster', 'Persen_Edukasi', 'Rasio_BL']].to_string(index=False))