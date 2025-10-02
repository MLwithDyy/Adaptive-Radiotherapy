# D.I.A.N.A (Dynamic Imaging-based Adaptive Neural Assistant)
An Intelligent System for Adaptive Radiotherapy (ART) in Cervical Cancer: Addressing Inter-Fraction Anatomical Variations using Machine Learning

# Predicting the Need for Adaptive Radiotherapy from Initial CT Scans

## 1. Project Objective

This research aims to develop a deep learning model capable of predicting whether a cervical cancer patient will require future adaptation of their radiation therapy plan by analyzing only the initial planning CT (Computed Tomography) images. The success of this model has the potential to become a clinical support tool for identifying high-risk patients early, enabling closer monitoring or adjustments to the therapy strategy.

---

## 2. Final Results

After undergoing a rigorous series of preprocessing, training, and validation processes, the developed pure 3D Convolutional Neural Network (CNN) model achieved very promising performance on a separate test dataset.

| **Key Metric** | **Score on Test Data** |
| ----------------------- | ---------------------- |
| **Accuracy** | **88.00%** |
| **Recall (Sensitivity)** | **100%** |
| **Precision** | 73%                    |
| **F1-Score** | 84%                    |

**Key Conclusion:** The model successfully **correctly identified all patients (100%)** who actually required therapy adaptation, which is the most critical clinical objective of this research.

---

## 3. Project Structure

This project consists of several Python scripts that must be run sequentially to replicate the results.


├── 01_convert_final.py      # (Local) Converts raw DICOM CT scans to NIfTI format.

├── 02_create_labels.py      # (Local) Automatically creates a label.txt file for each patient.

├── 03_split_dataset.py      # (Local) Splits the dataset into Training, Validation, & Test sets.

├── cnn_components_final.py  # Module containing the model architecture and data loader.

├── train_cnn_final.py        # (Colab) Main script for training the model using a GPU.

├── test_model_final.py       # (Colab) Script for testing the best model on the test data.

├── requirements.txt          # List of required Python libraries.

└── README.md                 # Explanation of this project.

---

## 4. Workflow & How to Run

### Step 0: Prerequisites

1.  **Initial Folder Structure:**
    * Create a main project folder on your computer, e.g., D:/AI_Project/.
    * Inside it, create a RAW_DATASET/ folder and place all patient DICOM data folders there (e.g., PT0, PT1, etc.).
    * Place the data_map.csv file (containing Patient_ID & Group) inside D:/AI_Project/.

2.  **Install Required Libraries:**
    Open a terminal or command prompt, navigate to your project folder, and run:
    ```bash
    pip install -r requirements.txt
    ```

### Step 1: Data Preparation (Run on a Local Machine/VS Code)

Run the following scripts sequentially. Important: Make sure to adjust the PATH variable at the top of each script to match your folder structure.

1.  **Convert DICOM to NIfTI:**
    ```bash
    python 01_convert_final.py
    ```
    * **Output:** It will create a PROCESSED_DATASET/ folder (or another name you specify) containing NIfTI files for each patient.

2.  **Create Label Files:**
    ```bash
    python 02_create_labels.py
    ```
    * **Output:** It will create a label.txt file inside each patient's subfolder in PROCESSED_DATASET/.

3.  **Split Dataset:**
    ```bash
    python 03_split_dataset.py
    ```
    * **Output:** It will create a final_data_map.csv file in your main folder.

4.  **Sync to Google Drive:**
    * Upload the entire PROCESSED_DATASET/ folder containing the NIfTI files and label.txt.
    * Upload the final_data_map.csv file.
    * Upload the cnn_components_final.py file.

### Step 2: Training & Testing (Run in Google Colab)

1.  **Prepare Colab Notebook:**
    * Create a new notebook in Google Colab.
    * Ensure you select a runtime with a GPU accelerator (Runtime -> Change runtime type -> GPU).

2.  **Train the Model:**
    * Copy and paste the entire content of the train_cnn_final.py script into a Colab notebook cell.
    * Ensure the DRIVE_BASE_PATH variable and other paths within the script match the location of your data in Google Drive.
    * Run the cell. This process will connect to your Drive, train the model, and save the best_model.h5 file to your Google Drive.

3.  **Test the Model:**
    * After training is complete, copy and paste the entire content of the test_model_final.py script into a new notebook cell.
    * Make sure the paths are correct.
    * Run the cell to get the final evaluation report and confusion matrix on the test data.
