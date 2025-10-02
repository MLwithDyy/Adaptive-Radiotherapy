import os
import pandas as pd

# --- KONFIGURASI (Disesuaikan agar sama dengan skrip lain) ---
# Path ke file CSV yang berisi daftar ID pasien dan grupnya
# Asumsi file ini berada di luar folder APLIKASI_NIFTI
PATH_PETA_DATA = "D:/D.I.A.N.A/Adaptive-Radiotherapy/peta_data_final_fix.csv" 

# Path ke folder utama yang akan berisi file label.txt
# Path ini SAMA DENGAN PATH_DATASET_OLAHAN di skrip statistik
PATH_DATASET_TUJUAN = "D:/DATASET/APLIKASI_NIFTI/"
# --- AKHIR KONFIGURASI ---

if __name__ == "__main__":
    # 1. Cek dan baca file peta data
    print(f"Membaca peta data dari: {PATH_PETA_DATA}")
    if not os.path.exists(PATH_PETA_DATA):
        print(f"Error: File peta data tidak ditemukan di '{PATH_PETA_DATA}'")
    else:
        df_peta = pd.read_csv(PATH_PETA_DATA)
        print(f"Ditemukan {len(df_peta)} total pasien.")

        # 2. Looping untuk membuat file label
        berhasil = 0
        gagal = 0
        print("\nMemulai proses pembuatan file label...")
        for index, row in df_peta.iterrows():
            id_pasien = str(row['ID_Pasien'])
            grup = row['Grup']
            
            # Tentukan path folder pasien di folder tujuan
            path_folder_pasien = os.path.join(PATH_DATASET_TUJUAN, id_pasien)
            
            try:
                # Buat folder jika belum ada
                os.makedirs(path_folder_pasien, exist_ok=True)
                
                # Tentukan label berdasarkan grup
                label = '1' if grup == 'Grup A' else '0'
                
                # Tentukan path file label
                path_file_label = os.path.join(path_folder_pasien, "label.txt")
                
                # Tulis label ke file
                with open(path_file_label, 'w') as f:
                    f.write(label)
                
                berhasil += 1
            except Exception as e:
                print(f"Gagal memproses {id_pasien}: {e}")
                gagal += 1
        
        print("\n" + "="*50)
        print("PROSES PEMBUATAN LABEL SELESAI")
        print(f"Berhasil: {berhasil} pasien")
        print(f"Gagal:    {gagal} pasien")
        print(f"File label.txt disimpan di dalam: {PATH_DATASET_TUJUAN}")
        print("="*50)

