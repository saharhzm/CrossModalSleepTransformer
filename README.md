# Cross-Modal SleepTransformer
Published in Journal of Biomedical Informatics
https://www.sciencedirect.com/science/article/abs/pii/S1532046424001072

![image](https://github.com/user-attachments/assets/d35e4031-bc01-4b3d-897c-9f3bf9a5f005)

**Overview**

The Cross-modal SleepTransformer is a deep learning model designed for sleep stage classification based on multimodal physiological signals. It utilizes a transformer architecture, and cross-attention mechanism to effectively process and integrate information from multiple input modalities, including EEG, EOG, EMG, ECG, and respiratory signals. By leveraging both raw signal data and handcrafted features extracted from these modalities, our model aims to accurately classify different sleep stages, including Wake, N1, N2, N3, and REM.


**Features**

- Integrates multiple physiological signals for sleep stage classification.
- Utilizes transformer encoder-decoder architecture for efficient information processing and integration.
- Supports both raw signal data and handcrafted features as input modalities.
- Provides flexibility in handling various types of physiological data.
- Facilitates accurate classification of different sleep stages.


**Requirements**

- Python 3.x
- TensorFlow 2.x
- NumPy
- Scipy
- Pandas
- Scikit-learn
- Imbalanced-learn
- Matplotlib



**Usage**

1. Prepare your dataset: Ensure your dataset is properly formatted and includes the necessary physiological signals (e.g., EEG, EOG, EMG, ECG).
2. Preprocess the data: Perform any required preprocessing steps, such as segmentation, and normalization on your dataset.
3. Train and test the model: Use the provided scripts to train and test the model on your dataset.

