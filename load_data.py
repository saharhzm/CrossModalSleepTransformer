import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from scipy import signal
from keras import regularizers
import keras_nlp
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten,BatchNormalization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from tensorflow.keras import layers
from utils import load_data, lr_schedule, split_and_normalize, pretrain_to_EDL_model
import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(True)
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from utils import load_data
import pandas as pd
import numpy as np
from sklearn import metrics
from collections import Counter

def split_and_normalize(all_data_dict, train_idx, test_idx):
    y_train = all_data_dict['SleepStage'][train_idx]
    x1 = all_data_dict['EEG']
    scaler = StandardScaler()
    x1 = scaler.fit_transform(x1)
    size = x1.shape[0]
    freqs, time, Sxx = signal.stft(x1, fs=125)
    spec_EEG1 = np.abs(Sxx)
    spec_EEG1_test = spec_EEG1[test_idx]
    x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
    x2 = all_data_dict['EEG(sec)']
    scaler = StandardScaler()
    x2 = scaler.fit_transform(x2)
    freqs, time, Sxx = signal.stft(x2,fs=125)
    spec_EEG2 = np.abs(Sxx)
    spec_EEG2_test = spec_EEG2[test_idx]
    x2 = x2.reshape(x2.shape[0],x2.shape[1],1)
    x = np.stack([x1,x2], axis=2)
    x_test_EEG = x[test_idx]
    x_test_EEG = x_test_EEG.reshape(x_test_EEG.shape[0],x_test_EEG.shape[1],x_test_EEG.shape[2])
    x1 = x1[train_idx]
    x2 = x2[train_idx]
    x1_aug = aug.jitter(x1)
    x2_aug = aug.jitter(x2)
    x3_aug = aug.magnitude_warp(x1_aug, 1)
    x4_aug = aug.magnitude_warp(x2_aug, 1)
    x5_aug = aug.window_slice(x3_aug, 1)
    x6_aug = aug.window_slice(x4_aug, 1)
    x1 = np.vstack([x1, x1_aug, x3_aug, x5_aug])
    x2 = np.vstack([x2, x2_aug, x4_aug, x6_aug])
    x_train_EEG = np.stack([x1,x2], axis=2)
    frames = [pd.DataFrame(y_train), pd.DataFrame(y_train), pd.DataFrame(y_train), pd.DataFrame(y_train)]
    y_train = pd.concat(frames)
    y_train = np.array(y_train)
    y_train = y_train.reshape(-1)
    shuffle_indices = np.random.permutation(len(y_train))
    y_train = np.array(y_train[shuffle_indices])
    x_train_EEG = x_train_EEG[shuffle_indices]
    x_train_EEG = x_train_EEG.reshape(x_train_EEG.shape[0],x_train_EEG.shape[1],x_train_EEG.shape[2])
    x1 = x1.reshape(x1.shape[0],x1.shape[1])
    x2 = x2.reshape(x2.shape[0],x2.shape[1])
    freqs, time, Sxx = signal.stft(x1, fs=125)
    spec_EEG1 = np.abs(Sxx)
    freqs, time, Sxx = signal.stft(x2,fs=125)
    spec_EEG2 = np.abs(Sxx)
    spec_EEG1_test = spec_EEG1_test.reshape(spec_EEG1_test.shape[0],spec_EEG1_test.shape[1],spec_EEG1_test.shape[2])
    print("x_train_EEG", x_train_EEG.shape)
    print("x_test_EEG", x_test_EEG.shape)
    spec_EEG1 = spec_EEG1.reshape(spec_EEG1.shape[0],spec_EEG1.shape[1],spec_EEG1.shape[2])
    print("spec_EEG1",spec_EEG1.shape)
    spec_EEG2 = spec_EEG2.reshape(spec_EEG2.shape[0],spec_EEG2.shape[1],spec_EEG2.shape[2])
    print("spec_EEG2",spec_EEG2.shape)
    spec_EEG1_test = spec_EEG1_test.reshape(spec_EEG1_test.shape[0],spec_EEG1_test.shape[1],spec_EEG1_test.shape[2])
    print("spec_EEG1 test",spec_EEG1_test.shape)
    spec_EEG2_test = spec_EEG2_test.reshape(spec_EEG2_test.shape[0],spec_EEG2_test.shape[1],spec_EEG2_test.shape[2])
    print("spec_EEG2 test",spec_EEG2_test.shape)

    y = Y[:size]
    y_train_HC = y[train_idx]
    y_test_HC = y[test_idx]
    
    scaler = StandardScaler()
    x = scaler.fit_transform(handcraft_EEG)
    x = x[:size]
    print("HC size", x.shape)
    EEG_HC = x[train_idx]
    EEG_HC = EEG_HC.reshape(EEG_HC.shape[0], EEG_HC.shape[1], 1)
    EEG_HC_aug1 = aug.jitter(EEG_HC)
    EEG_HC_aug2 = aug.magnitude_warp(EEG_HC_aug1, 1)
    EEG_HC_aug3 = aug.window_slice(EEG_HC_aug2, 1)
    EEG_HC = np.vstack([EEG_HC, EEG_HC_aug1, EEG_HC_aug2, EEG_HC_aug3])
    EEG_HC = EEG_HC[shuffle_indices]
    EEG_HC_test = x[test_idx]
    EEG_HC_test = EEG_HC_test.reshape(EEG_HC_test.shape[0], EEG_HC_test.shape[1], 1)
    print("EEG_HC",EEG_HC.shape)
    print("EEG_HC_test",EEG_HC_test.shape)
    print("**********************")

    x = all_data_dict['EMG']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = x.reshape(x.shape[0],x.shape[1],1)
    x_train_EMG = x[train_idx]
    x_train_EMG_aug1 = aug.jitter(x_train_EMG)
    x_train_EMG_aug2 = aug.magnitude_warp(x_train_EMG_aug1, 1)
    x_train_EMG_aug3 = aug.window_slice(x_train_EMG_aug2, 1)
    x_train_EMG = np.vstack([x_train_EMG, x_train_EMG_aug1, x_train_EMG_aug2, x_train_EMG_aug3])
    x_train_EMG = x_train_EMG[shuffle_indices]
    x_test_EMG = x[test_idx]
    x_train_EMG = x_train_EMG.reshape(x_train_EMG.shape[0],x_train_EMG.shape[1])
    x_test_EMG = x_test_EMG.reshape(x_test_EMG.shape[0],x_test_EMG.shape[1])
    print("x_train_EMG", x_train_EMG.shape)
    print("x_test_EMG", x_test_EMG.shape)
    freqs, time, Sxx = signal.stft(x_test_EMG, fs=125)
    spec_EMG_test = np.abs(Sxx)
    print("spec_EMG",spec_EMG_test.shape)
    freqs, time, Sxx = signal.stft(x_train_EMG, fs=125)
    spec_EMG = np.abs(Sxx)
    print("spec_EMG",spec_EMG.shape)
    
    scaler = StandardScaler()
    x = scaler.fit_transform(handcraft_EMG)
    x = x[:size]
    print("HC", x.shape)
    EMG_HC = x[train_idx]
    EMG_HC = EMG_HC.reshape(EMG_HC.shape[0], EMG_HC.shape[1], 1)
    EMG_HC_aug1 = aug.jitter(EMG_HC)
    EMG_HC_aug2 = aug.magnitude_warp(EMG_HC_aug1, 1)
    EMG_HC_aug3 = aug.window_slice(EMG_HC_aug2, 1)
    EMG_HC = np.vstack([EMG_HC, EMG_HC_aug1, EMG_HC_aug2, EMG_HC_aug3])
    EMG_HC = EMG_HC[shuffle_indices]
    EMG_HC_test = x[test_idx]
    EMG_HC_test = EMG_HC_test.reshape(EMG_HC_test.shape[0], EMG_HC_test.shape[1], 1)
    print("EMG_HC",EMG_HC.shape)
    print("EMG_HC_test",EMG_HC_test.shape)
    print("**********************")

    x = all_data_dict['ECG']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = x.reshape(x.shape[0],x.shape[1],1)
    x_train_ECG = x[train_idx]
    x_train_ECG_aug1 = aug.jitter(x_train_ECG)
    x_train_ECG_aug2 = aug.magnitude_warp(x_train_ECG_aug1, 1)
    x_train_ECG_aug3 = aug.window_slice(x_train_ECG_aug2, 1)
    x_train_ECG = np.vstack([x_train_ECG, x_train_ECG_aug1, x_train_ECG_aug2, x_train_ECG_aug3])
    x_train_ECG = x_train_ECG[shuffle_indices]
    x_train_ECG = x_train_ECG.reshape(x_train_ECG.shape[0],x_train_ECG.shape[1])
    x_test_ECG = x[test_idx]
    x_test_ECG = x_test_ECG.reshape(x_test_ECG.shape[0],x_test_ECG.shape[1])
    print("x_train_ECG", x_train_ECG.shape)
    print("x_test_ECG", x_test_ECG.shape)
    freqs, time, Sxx = signal.stft(x_train_ECG, fs=125)
    spec_ECG = np.abs(Sxx)
    print("spec_ECG",spec_ECG.shape)
    freqs, time, Sxx = signal.stft(x_test_ECG, fs=125)
    spec_ECG_test = np.abs(Sxx)
    print("spec_ECG",spec_ECG_test.shape)
    

    scaler = StandardScaler()
    x = scaler.fit_transform(handcraft_ECG)
    x = x[:size]
    print("HC", x.shape)
    ECG_HC = x[train_idx]
    ECG_HC = ECG_HC.reshape(ECG_HC.shape[0], ECG_HC.shape[1], 1)
    ECG_HC_aug1 = aug.jitter(ECG_HC)
    ECG_HC_aug2 = aug.magnitude_warp(ECG_HC_aug1, 1)
    ECG_HC_aug3 = aug.window_slice(ECG_HC_aug2, 1)
    ECG_HC = np.vstack([ECG_HC, ECG_HC_aug1, ECG_HC_aug2, ECG_HC_aug3])
    ECG_HC = ECG_HC[shuffle_indices]
    ECG_HC_test = x[test_idx]
    ECG_HC_test = ECG_HC_test.reshape(ECG_HC_test.shape[0], ECG_HC_test.shape[1], 1)
    print("ECG_HC",ECG_HC.shape)
    print("ECG_HC_test",ECG_HC_test.shape)
    print("**********************")

    x1 = all_data_dict['EOG(L)']
    scaler = StandardScaler()
    x1 = scaler.fit_transform(x1)
    freqs, time, Sxx = signal.stft(x1, fs=50)
    spec_EOG1 = np.abs(Sxx)
    spec_EOG1_test = spec_EOG1[test_idx]
    x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
    x2 = all_data_dict['EOG(R)']
    scaler = StandardScaler()
    x2 = scaler.fit_transform(x2)
    freqs, time, Sxx = signal.stft(x2,fs=50)
    spec_EOG2 = np.abs(Sxx)
    spec_EOG2_test = spec_EOG2[test_idx]
    x2 = x2.reshape(x2.shape[0],x2.shape[1],1)
    x = np.stack([x1,x2], axis=2)
    x_test_EOG = x[test_idx]
    x_test_EOG = x_test_EOG.reshape(x_test_EOG.shape[0],x_test_EOG.shape[1],x_test_EOG.shape[2])
    x1 = x1[train_idx]
    x2 = x2[train_idx]
    x1_aug = aug.jitter(x1)
    x2_aug = aug.jitter(x2)
    x3_aug = aug.magnitude_warp(x1_aug, 1)
    x4_aug = aug.magnitude_warp(x2_aug, 1)
    x5_aug = aug.window_slice(x3_aug, 1)
    x6_aug = aug.window_slice(x4_aug, 1)
    x1 = np.vstack([x1, x1_aug, x3_aug, x5_aug])
    x2 = np.vstack([x2, x2_aug, x4_aug, x6_aug])
    x_train_EOG = np.stack([x1,x2], axis=2)
    x_train_EOG = x_train_EOG[shuffle_indices]
    x1 = x1.reshape(x1.shape[0],x1.shape[1])
    x2 = x2.reshape(x2.shape[0],x2.shape[1])
    freqs, time, Sxx = signal.stft(x1, fs=50)
    spec_EOG1 = np.abs(Sxx)
    freqs, time, Sxx = signal.stft(x2,fs=50)
    spec_EOG2 = np.abs(Sxx)
    x_train_EOG = x_train_EOG.reshape(x_train_EOG.shape[0],x_train_EOG.shape[1],x_train_EOG.shape[2])
    print("x_train_EOG", x_train_EOG.shape)
    print("x_test_EOG", x_test_EOG.shape)
    spec_EOG1 = spec_EOG1.reshape(spec_EOG1.shape[0],spec_EOG1.shape[1],spec_EOG1.shape[2])
    print("spec_EOG1",spec_EOG2.shape)
    spec_EOG2 = spec_EOG2.reshape(spec_EOG2.shape[0],spec_EOG2.shape[1],spec_EOG2.shape[2])
    print("spec_EOG2",spec_EOG2.shape)
    spec_EOG1_test = spec_EOG1_test.reshape(spec_EOG1_test.shape[0],spec_EOG1_test.shape[1],spec_EOG1_test.shape[2])
    print("spec_EOG1_test",spec_EOG1_test.shape)
    spec_EOG2_test = spec_EOG2_test.reshape(spec_EOG2_test.shape[0],spec_EOG2_test.shape[1],spec_EOG2_test.shape[2])
    print("spec_EOG2_test",spec_EOG2_test.shape)
    
    scaler = StandardScaler()
    x = scaler.fit_transform(handcraft_EOG)
    x = x[:size]
    print("HC", x.shape)
    EOG_HC = x[train_idx]
    EOG_HC = EOG_HC.reshape(EOG_HC.shape[0], EOG_HC.shape[1], 1)
    EOG_HC_aug1 = aug.jitter(EOG_HC)
    EOG_HC_aug2 = aug.magnitude_warp(EOG_HC_aug1, 1)
    EOG_HC_aug3 = aug.window_slice(EOG_HC_aug2, 1)
    EOG_HC = np.vstack([EOG_HC, EOG_HC_aug1, EOG_HC_aug2, EOG_HC_aug3])
    EOG_HC = EOG_HC[shuffle_indices]
    EOG_HC_test = x[test_idx]
    EOG_HC_test = EOG_HC_test.reshape(EOG_HC_test.shape[0], EOG_HC_test.shape[1], 1)
    print("EOG_HC",EOG_HC.shape)
    print("EOG_HC_test",EOG_HC_test.shape)
    print("**********************")

    x = all_data_dict['NEW AIR']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = x.reshape(x.shape[0],x.shape[1],1)
    x_train_NEWAIR = x[train_idx]
    x_train_NEWAIR_aug1 = aug.jitter(x_train_NEWAIR)
    x_train_NEWAIR_aug2 = aug.magnitude_warp(x_train_NEWAIR_aug1, 1)
    x_train_NEWAIR_aug3 = aug.window_slice(x_train_NEWAIR_aug2, 1)
    x_train_NEWAIR = np.vstack([x_train_NEWAIR, x_train_NEWAIR_aug1, x_train_NEWAIR_aug2, x_train_NEWAIR_aug3])
    x_train_NEWAIR = x_train_NEWAIR[shuffle_indices]
    x_train_NEWAIR = x_train_NEWAIR.reshape(x_train_NEWAIR.shape[0],x_train_NEWAIR.shape[1])
    x_test_NEWAIR = x[test_idx]
    x_test_NEWAIR = x_test_NEWAIR.reshape(x_test_NEWAIR.shape[0],x_test_NEWAIR.shape[1])
    print("x_train_NEWAIR", x_train_NEWAIR.shape)
    print("x_test_NEWAIR", x_test_NEWAIR.shape)
    freqs, time, Sxx = signal.stft(x_train_NEWAIR, fs=10)
    spec_NEWAIR = np.abs(Sxx)
    print("spec_NEWAIR",spec_NEWAIR.shape)
    freqs, time, Sxx = signal.stft(x_test_NEWAIR, fs=10)
    spec_NEWAIR_test = np.abs(Sxx)
    print("spec_NEWAIR",spec_NEWAIR_test.shape)
    
    scaler = StandardScaler()
    x = scaler.fit_transform(handcraft_NEWAIR)
    x = x[:size]
    print("HC", x.shape)
    NEWAIR_HC = x[train_idx]
    NEWAIR_HC = NEWAIR_HC.reshape(NEWAIR_HC.shape[0], NEWAIR_HC.shape[1], 1)
    NEWAIR_HC_aug1 = aug.jitter(NEWAIR_HC)
    NEWAIR_HC_aug2 = aug.magnitude_warp(NEWAIR_HC_aug1, 1)
    NEWAIR_HC_aug3 = aug.window_slice(NEWAIR_HC_aug2, 1)
    NEWAIR_HC = np.vstack([NEWAIR_HC, NEWAIR_HC_aug1, NEWAIR_HC_aug2, NEWAIR_HC_aug3])
    NEWAIR_HC = NEWAIR_HC[shuffle_indices]
    NEWAIR_HC_test = x[test_idx]
    NEWAIR_HC_test = NEWAIR_HC_test.reshape(NEWAIR_HC_test.shape[0], NEWAIR_HC_test.shape[1], 1)
    print("NEWAIR_HC",NEWAIR_HC.shape)
    print("NEWAIR_HC_test",NEWAIR_HC_test.shape)
    print("**********************")

    x = all_data_dict['THOR RES']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = x.reshape(x.shape[0],x.shape[1],1)
    x_train_THOR = x[train_idx]
    x_train_THOR_aug1 = aug.jitter(x_train_THOR)
    x_train_THOR_aug2 = aug.magnitude_warp(x_train_THOR_aug1, 1)
    x_train_THOR_aug3 = aug.window_slice(x_train_THOR_aug2, 1)
    x_train_THOR = np.vstack([x_train_THOR, x_train_THOR_aug1, x_train_THOR_aug2, x_train_THOR_aug3])
    x_train_THOR = x_train_THOR[shuffle_indices]
    x_train_THOR = x_train_THOR.reshape(x_train_THOR.shape[0],x_train_THOR.shape[1])
    x_test_THOR = x[test_idx]
    x_test_THOR = x_test_THOR.reshape(x_test_THOR.shape[0],x_test_THOR.shape[1])
    print("x_train_THOR", x_train_THOR.shape)
    print("x_test_THOR", x_test_THOR.shape)
    freqs, time, Sxx = signal.stft(x_train_THOR, fs=10)
    spec_THOR = np.abs(Sxx)
    print("spec_THOR",spec_THOR.shape)
    freqs, time, Sxx = signal.stft(x_test_THOR, fs=10)
    spec_THOR_test = np.abs(Sxx)
    print("spec_THOR",spec_THOR_test.shape)

    scaler = StandardScaler()
    x = scaler.fit_transform(handcraft_THOR)
    x = x[:size]
    print("HC", x.shape)
    THOR_HC = x[train_idx]
    THOR_HC = THOR_HC.reshape(THOR_HC.shape[0], THOR_HC.shape[1], 1)
    THOR_HC_aug1 = aug.jitter(THOR_HC)
    THOR_HC_aug2 = aug.magnitude_warp(THOR_HC_aug1, 1)
    THOR_HC_aug3 = aug.window_slice(THOR_HC_aug2, 1)
    THOR_HC = np.vstack([THOR_HC, THOR_HC_aug1, THOR_HC_aug2, THOR_HC_aug3])
    THOR_HC = THOR_HC[shuffle_indices]
    THOR_HC_test = x[test_idx]
    THOR_HC_test = THOR_HC_test.reshape(THOR_HC_test.shape[0], THOR_HC_test.shape[1], 1)
    print("THOR_HC",THOR_HC.shape)
    print("THOR_HC_test",THOR_HC_test.shape)
    print("**********************")

    x = all_data_dict['ABDO RES']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = x.reshape(x.shape[0],x.shape[1],1)
    x_train_ABDO = x[train_idx]
    x_train_ABDO_aug1 = aug.jitter(x_train_ABDO)
    x_train_ABDO_aug2 = aug.magnitude_warp(x_train_ABDO_aug1, 1)
    x_train_ABDO_aug3 = aug.window_slice(x_train_ABDO_aug2, 1)
    x_train_ABDO = np.vstack([x_train_ABDO, x_train_ABDO_aug1, x_train_ABDO_aug2, x_train_ABDO_aug3])
    x_train_ABDO = x_train_ABDO[shuffle_indices]
    x_train_ABDO = x_train_ABDO.reshape(x_train_ABDO.shape[0],x_train_ABDO.shape[1])
    x_test_ABDO = x[test_idx]
    x_test_ABDO = x_test_ABDO.reshape(x_test_ABDO.shape[0],x_test_ABDO.shape[1])
    print("x_train_ABDO", x_train_ABDO.shape)
    print("x_test_ABDO", x_test_ABDO.shape)
    freqs, time, Sxx = signal.stft(x_train_ABDO, fs=10)
    spec_ABDO = np.abs(Sxx)
    print("spec_ABDO",spec_ABDO.shape)
    freqs, time, Sxx = signal.stft(x_test_ABDO, fs=10)
    spec_ABDO_test = np.abs(Sxx)
    print("spec_ABDO_test",spec_ABDO_test.shape)
    
    scaler = StandardScaler()
    x = scaler.fit_transform(handcraft_ABDO)
    x = x[:size]
    print("HC", x.shape)
    ABDO_HC = x[train_idx]
    ABDO_HC = ABDO_HC.reshape(ABDO_HC.shape[0], ABDO_HC.shape[1], 1)
    ABDO_HC_aug1 = aug.jitter(ABDO_HC)
    ABDO_HC_aug2 = aug.magnitude_warp(ABDO_HC_aug1, 1)
    ABDO_HC_aug3 = aug.window_slice(ABDO_HC_aug2, 1)
    ABDO_HC = np.vstack([ABDO_HC, ABDO_HC_aug1, ABDO_HC_aug2, ABDO_HC_aug3])
    ABDO_HC = ABDO_HC[shuffle_indices]
    ABDO_HC_test = x[test_idx]
    ABDO_HC_test = ABDO_HC_test.reshape(ABDO_HC_test.shape[0], ABDO_HC_test.shape[1], 1)
    print("ABDO_HC",ABDO_HC.shape)
    print("ABDO_HC_test",ABDO_HC_test.shape)
    print("**********************")

    return y_train, x_train_EEG, x_test_EEG, x_train_EMG, x_test_EMG, x_train_ECG, x_test_ECG, x_train_EOG, x_test_EOG, x_train_NEWAIR, x_test_NEWAIR, x_train_THOR, x_test_THOR, x_train_ABDO, x_test_ABDO, spec_EEG1, spec_EEG2, spec_EMG, spec_ECG, spec_EOG1, spec_EOG2, spec_NEWAIR, spec_THOR, spec_ABDO, spec_EEG1_test, spec_EEG2_test, spec_EMG_test, spec_ECG_test, spec_EOG1_test, spec_EOG2_test, spec_NEWAIR_test, spec_THOR_test, spec_ABDO_test, EEG_HC, EEG_HC_test, EMG_HC, EMG_HC_test, ECG_HC, ECG_HC_test, EOG_HC, EOG_HC_test, NEWAIR_HC, NEWAIR_HC_test, THOR_HC, THOR_HC_test, ABDO_HC, ABDO_HC_test
