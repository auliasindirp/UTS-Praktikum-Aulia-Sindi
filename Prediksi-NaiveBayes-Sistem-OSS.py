# ============================================================
#  PREDIKSI JENIS PROYEK MENGGUNAKAN NAÏVE BAYES + SMOTE
#  Dataset: DP.Proyek1 (1).xlsx (Sheet2)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 1️⃣ LOAD DATASET
# ============================================================
data = pd.read_excel("DP.Proyek1 (1).xlsx", sheet_name="Sheet2")

print("=== DATA AWAL ===")
print(data.head())
print("\nJumlah baris dan kolom:", data.shape)

# ============================================================
# 2️⃣ PEMBERSIHAN DATA
# ============================================================
# Hapus baris kosong pada kolom penting
data = data.dropna(subset=['Jumlah Investasi', 'luastanah', 'TKI', 'UraianJenisProyek'])

# Bersihkan simbol non-numerik di kolom angka
def bersihkan_angka(val):
    if isinstance(val, str):
        val = val.replace('Rp', '').replace('.', '').replace(',', '').strip()
    return pd.to_numeric(val, errors='coerce')

data['Jumlah Investasi'] = data['Jumlah Investasi'].apply(bersihkan_angka)
data['luastanah'] = data['luastanah'].apply(bersihkan_angka)
data['TKI'] = data['TKI'].apply(bersihkan_angka)

# Hapus baris kosong setelah konversi
data = data.dropna(subset=['Jumlah Investasi', 'luastanah', 'TKI'])

# ============================================================
# 3️⃣ PILIH FITUR DAN LABEL
# ============================================================
X = data[['Jumlah Investasi', 'luastanah', 'TKI']]
y = data['UraianJenisProyek']

# Encode label (Utama / Pendukung)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Tampilkan distribusi awal
print("\n=== Distribusi Sebelum SMOTE ===")
print(pd.Series(y_encoded).value_counts())

# ============================================================
# 4️⃣ BALANCING DATA DENGAN SMOTE
# ============================================================
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y_encoded)

# Cek distribusi setelah SMOTE
print("\n=== Distribusi Setelah SMOTE ===")
print(pd.Series(y_res).value_counts())

# ============================================================
# 5️⃣ NORMALISASI FITUR
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# ============================================================
# 6️⃣ SPLIT DATA (TRAIN & TEST)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, random_state=42
)

print("\nData latih:", X_train.shape)
print("Data uji:", X_test.shape)

# ============================================================
# 7️⃣ LATIH MODEL NAÏVE BAYES
# ============================================================
model = GaussianNB()
model.fit(X_train, y_train)
print("\nModel berhasil dilatih ✅")

# ============================================================
# 8️⃣ PREDIKSI DAN EVALUASI
# ============================================================
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred) * 100

print("\n=== HASIL EVALUASI ===")
print(f"Akurasi model: {akurasi:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix – Naïve Bayes dengan SMOTE')
plt.tight_layout()
plt.show()

# ============================================================
# 9️⃣ PREDIKSI MANUAL
# ============================================================
try:
    inv = float(input("\nMasukkan jumlah investasi (Rp): "))
    tanah = float(input("Masukkan luas tanah (m²): "))
    tki = float(input("Masukkan jumlah TKI: "))

    input_scaled = scaler.transform([[inv, tanah, tki]])
    pred = model.predict(input_scaled)
    print(f"\n🔍 Jenis proyek diprediksi sebagai: {le.inverse_transform(pred)[0]}")
except:
    print("\nInput manual dilewati.")