import os
import pydicom
import SimpleITK as sitk
from collections import defaultdict

# --- KONFIGURASI (PENTING: SESUAIKAN PATH INI!) ---
# Ganti dengan path di komputer Anda. Gunakan forward slash '/' untuk kompatibilitas.
PATH_DATASET_MENTAH = "D:/DATASET/APLIKASI/"
PATH_DATASET_OLAHAN = "D:/DATASET/APLIKASI_NIFTI/"
# --- AKHIR KONFIGURASI ---


def resample_image_to_uniform_spacing(image, target_spacing=(1.0, 1.0, 2.5), interpolator=sitk.sitkLinear):
    """
    Fungsi untuk me-resample sebuah gambar ke spasi baru yang seragam.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(interpolator)
    
    return resampler.Execute(image)


def find_main_ct_series(patient_folder):
    """
    Fungsi cerdas untuk menemukan dan MENGURUTKAN seri CT utama.
    """
    file_list = [os.path.join(root, name) for root, _, files in os.walk(patient_folder) for name in files]
    
    series_info = defaultdict(list)
    for file_path in file_list:
        try:
            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
            if "SeriesInstanceUID" in dcm:
                series_info[dcm.SeriesInstanceUID].append(file_path)
        except Exception:
            continue

    if not series_info: return None

    ct_series_candidates = []
    for series_uid, filenames in series_info.items():
        first_file_header = pydicom.dcmread(filenames[0], stop_before_pixels=True)
        if first_file_header.get("Modality", "").upper() == 'CT':
            ct_series_candidates.append(filenames)
            
    if not ct_series_candidates: return None
        
    ct_series_candidates.sort(key=len, reverse=True)
    main_series_files = ct_series_candidates[0]

    # --- LANGKAH SORTING BARU & KRUSIAL ---
    slice_locations = []
    for filename in main_series_files:
        try:
            dcm = pydicom.dcmread(filename, stop_before_pixels=True)
            z_pos = float(dcm.ImagePositionPatient[2])
            slice_locations.append((filename, z_pos))
        except Exception:
            continue
            
    # --- PERUBAHAN DI SINI: reverse=True ---
    # Sortir daftar file berdasarkan posisi z secara terbalik (dari besar ke kecil)
    # untuk memperbaiki masalah orientasi atas-bawah.
    slice_locations.sort(key=lambda x: x[1], reverse=True)

    sorted_filenames = [item[0] for item in slice_locations]
    
    return sorted_filenames


if __name__ == "__main__":
    print(f"Memindai folder dataset mentah di: {PATH_DATASET_MENTAH}")
    if not os.path.isdir(PATH_DATASET_MENTAH):
        print(f"\nERROR: Folder dataset mentah tidak ditemukan di '{PATH_DATASET_MENTAH}'.")
    else:
        patient_ids = [pid for pid in os.listdir(PATH_DATASET_MENTAH) if os.path.isdir(os.path.join(PATH_DATASET_MENTAH, pid))]
        print(f"Ditemukan {len(patient_ids)} folder pasien untuk diproses.")

        berhasil_count, gagal_count = 0, 0
        print("\nMemulai proses konversi DICOM ke NIfTI...")
        for pid in patient_ids:
            print(f"--- Memproses Pasien: {pid} ---")
            
            path_folder_pasien_mentah = os.path.join(PATH_DATASET_MENTAH, pid)
            path_folder_output_pasien = os.path.join(PATH_DATASET_OLAHAN, pid)
            path_file_output_nifti = os.path.join(path_folder_output_pasien, f"{pid}_AWAL.nii.gz")
            
            try:
                os.makedirs(path_folder_output_pasien, exist_ok=True)

                sorted_ct_filenames = find_main_ct_series(path_folder_pasien_mentah)
                
                if not sorted_ct_filenames:
                    raise RuntimeError("Tidak ada seri citra CT yang valid ditemukan.")
                
                print(f"   -> Seri CT utama ditemukan dan diurutkan ({len(sorted_ct_filenames)} irisan). Membaca data...")

                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(sorted_ct_filenames)
                image_3d = reader.Execute()
                
                print(f"   -> Melakukan resampling untuk menstandarkan spasi voxel...")
                resampled_image = resample_image_to_uniform_spacing(image_3d)
                
                print(f"   -> Menyimpan hasil ke: {path_file_output_nifti}")
                sitk.WriteImage(resampled_image, path_file_output_nifti)
                
                print(f"   -> Konversi untuk {pid} berhasil.")
                berhasil_count += 1

            except Exception as e:
                print(f"   -> ERROR: Terjadi kesalahan saat memproses {pid}: {e}")
                gagal_count += 1

        print("\n" + "="*50)
        print("PROSES KONVERSI SELESAI")
        print(f"Berhasil: {berhasil_count} | Gagal/Dilewati: {gagal_count}")
        print("="*50)

