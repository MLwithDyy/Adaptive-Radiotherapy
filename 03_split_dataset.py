import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- KONFIGURASI ---
PATH_PETA_DATA = "D:/DATASET/peta_data.csv"
PATH_OUTPUT_CSV = "D:/DATASET/peta_data_final.csv"
UKURAN_UJI = 0.15
UKURAN_VALIDASI = 0.15
# --------------------

if __name__ == "__main__":
    if not os.path.exists(PATH_PETA_DATA):
        print(f"Error: File peta data tidak ditemukan di '{PATH_PETA_DATA}'")
    else:
        df = pd.read_csv(PATH_PETA_DATA)

        X = df['ID_Pasien']
        y = df['Grup']

        X_sisa, X_uji, y_sisa, y_uji = train_test_split(
            X, y, test_size=UKURAN_UJI, random_state=42, stratify=y
        )
        X_latih, X_validasi, y_latih, y_validasi = train_test_split(
            X_sisa, y_sisa, test_size=UKURAN_VALIDASI, random_state=42, stratify=y_sisa
        )

        df['Set'] = ''
        df.loc[X_latih.index, 'Set'] = 'Latih'
        df.loc[X_validasi.index, 'Set'] = 'Validasi'
        df.loc[X_uji.index, 'Set'] = 'Uji'

        df.to_csv(PATH_OUTPUT_CSV, index=False)

        print("\nPEMBAGIAN DATASET SELESAI")
        print(f"Total: {len(df)} | Latih: {len(X_latih)} | Validasi: {len(X_validasi)} | Uji: {len(X_uji)}")
        print(f"Peta data final disimpan di: {PATH_OUTPUT_CSV}")
