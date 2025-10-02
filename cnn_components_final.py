import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom, rotate

# --- BAGIAN 1: ARSITEKTUR MODEL CNN MURNI ---

def build_pure_cnn_model(input_shape_gambar):
    """Membangun model klasifikasi CNN 3D murni."""
    
    input_gambar = Input(shape=input_shape_gambar, name="input_citra_awal")
    
    # Menggunakan arsitektur 'Lite' untuk mencegah overfitting
    x = Conv3D(8, 3, padding="same", activation="relu")(input_gambar)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(16, 3, padding="same", activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(32, 3, padding="same", activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    
    # Prediction Head
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid', name="output_prediksi")(x)

    model = Model(inputs=input_gambar, outputs=output_layer, name="model_cnn_murni")
    return model

# --- BAGIAN 2: DATA LOADER (Hanya Gambar) ---

def _load_and_process_image_py(citra_path, label, ukuran_target, augment):
    """Fungsi Python untuk memuat satu sampel gambar."""
    citra_path = citra_path.numpy().decode('utf-8')
    label = label.numpy()
    ukuran_target = ukuran_target.numpy()
    augment = augment.numpy()
    
    img = sitk.ReadImage(citra_path, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)
    faktor_zoom = [
        ukuran_target[2] / arr.shape[0],
        ukuran_target[1] / arr.shape[1],
        ukuran_target[0] / arr.shape[2]
    ]
    arr_resized = zoom(arr, faktor_zoom, order=1)
    arr_final = np.transpose(arr_resized, (2, 1, 0))
    citra = arr_final[:, :, :, np.newaxis].astype(np.float32)

    if augment:
        if np.random.rand() > 0.5:
            citra = np.flip(citra, axis=np.random.choice([0, 1, 2])).copy()
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-10, 10)
            axes = tuple(np.random.choice([0, 1, 2], 2, replace=False))
            citra = rotate(citra, angle, axes=axes, reshape=False, order=1, cval=0).copy()

    return citra, np.array([label], dtype=np.float32)

def create_image_only_dataset(patient_list, data_dir, batch_size, ukuran_gambar, augment=False):
    """Fungsi utama untuk membuat tf.data.Dataset hanya dari gambar."""
    citra_files, labels = [], []
    for pid, grup in patient_list:
        path_citra = os.path.join(data_dir, pid, f"{pid}_AWAL.nii.gz")
        if os.path.exists(path_citra):
            citra_files.append(path_citra)
            labels.append(1 if grup == 'Grup A' else 0)

    dataset = tf.data.Dataset.from_tensor_slices((citra_files, labels))
    
    def _map_fn(citra_path, label):
        ukuran_tensor = tf.constant(ukuran_gambar, dtype=tf.int64)
        augment_tensor = tf.constant(augment, dtype=tf.bool)
        citra, label_out = tf.py_function(
            _load_and_process_image_py,
            inp=[citra_path, label, ukuran_tensor, augment_tensor],
            Tout=[tf.float32, tf.float32]
        )
        citra.set_shape(list(ukuran_gambar))
        label_out.set_shape([1])
        return citra, label_out

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.shuffle(buffer_size=len(citra_files))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
