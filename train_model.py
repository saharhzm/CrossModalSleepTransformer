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
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from utils import load_data
import pandas as pd
import numpy as np
from sklearn import metrics
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import pywt
from keras.layers import Bidirectional
from keras.layers import LSTM, GRU
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from keras import backend as K
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHead
import augmentation as aug
from keras.layers import Layer
import tensorflow_probability as tfp
from sklearn.metrics import roc_auc_score
from cross_modal_transformer import CrossModalTransformer
from cnn_attention_blocks import extractor_10 ,extractor_50, extractor_125, extractor_hc, extractor_spec

print("load data...")
dict_data_dir = "data/segmented_data/"
all_data_dict = load_data(dict_data_dir)
data_dir = 'D:/SleepStaging/data/expert_features/feature v1/50 with ECG EEG/50 sub part 3 Concat EEG1,2 EOG 1,2/'
handcraft_EEG = np.load(data_dir+'X_EEG.npy',allow_pickle=True)
handcraft_EMG = np.load(data_dir+'X_EMG.npy',allow_pickle=True)
handcraft_ECG = np.load(data_dir+'X_ECG.npy',allow_pickle=True)
handcraft_EOG = np.load(data_dir+'X_EOG.npy',allow_pickle=True)
handcraft_NEWAIR = np.load(data_dir+'X_NEWAIR.npy',allow_pickle=True)
handcraft_THOR = np.load(data_dir+'X_THORRES.npy',allow_pickle=True)
handcraft_ABDO = np.load(data_dir+'X_ABDORES.npy',allow_pickle=True)
Y = np.load(data_dir+'Y_all.npy',allow_pickle=True)

def SleepStage():
    
    print("SleepStage")
    
    input_EEG = keras.Input(shape=(3750,2))
    input_EMG = keras.Input(shape=(3750,1))
    input_ECG = keras.Input(shape=(3750,1))
    input_EOG = keras.Input(shape=(1500,2))
    input_NEWAIR = keras.Input(shape=(300,1))
    input_THOR = keras.Input(shape=(300,1))
    input_ABDO = keras.Input(shape=(300,1))

    input_HC_EEG = keras.Input(shape=(344,1), name='HC_EEG')
    input_HC_EMG = keras.Input(shape=(160,1), name='HC_EMG')
    input_HC_ECG = keras.Input(shape=(169,1), name='HC_ECG')
    input_HC_EOG = keras.Input(shape=(256,1), name='HC_EOG')
    input_HC_NEWAIR = keras.Input(shape=(96,1), name='HC_NEWAIR')
    input_HC_THOR = keras.Input(shape=(96,1), name='HC_THOR')
    input_HC_ABDO = keras.Input(shape=(96,1), name='HC_ABDO')

    x_EEG = extractor_125(input_EEG, fs=125, name='EEG')
    x_EMG = extractor_125(input_EMG, fs=125, name='EMG')
    x_ECG = extractor_125(input_ECG, fs=125, name='ECG')
    x_EOG = extractor_50(input_EOG, fs=50, name='EOG')
    x_NEWAIR = extractor_10(input_NEWAIR, fs=10, name='NEWAIR')
    x_THOR = extractor_10(input_THOR, fs=10, name='THOR')
    x_ABDO = extractor_10(input_ABDO, fs=10, name='ABDO')
    
    print("features")
    print(x_EEG.shape)
    print(x_EMG.shape)
    print(x_ECG.shape)
    print(x_EOG.shape)
    print(x_NEWAIR.shape)
    print(x_THOR.shape)
    print(x_ABDO.shape)

    EEG_HC = extractor_CNN_HC(input_HC_EEG, fs=125, name='HC')
    EMG_HC = extractor_CNN_HC(input_HC_EMG, fs=125, name='HC')
    ECG_HC = extractor_CNN_HC(input_HC_ECG, fs=125, name='HC')
    EOG_HC = extractor_CNN_HC(input_HC_EOG, fs=125, name='HC')
    NEWAIR_HC = extractor_CNN_HC(input_HC_NEWAIR, fs=125, name='HC')
    THOR_HC = extractor_CNN_HC(input_HC_THOR, fs=125, name='HC')
    ABDO_HC = extractor_CNN_HC(input_HC_ABDO, fs=125, name='HC')
    
    print("HC features")
    print(EEG_HC.shape)
    print(EMG_HC.shape)
    print(ECG_HC.shape)
    print(EOG_HC.shape)
    print(NEWAIR_HC.shape)
    print(THOR_HC.shape)
    print(ABDO_HC.shape)

    features1 = layers.Concatenate(axis=1)([x_EEG, x_EMG, x_ECG, x_EOG, x_NEWAIR, x_THOR, x_ABDO])
    features2 = layers.Concatenate(axis=1)([EEG_HC, EMG_HC, ECG_HC, EOG_HC, NEWAIR_HC, THOR_HC, ABDO_HC])
    
    transformer_inputs1 = layers.Dense(448, activation="relu")(features1)
    transformer_inputs1 = layers.Reshape((7, 64), input_shape=transformer_inputs1.shape)(transformer_inputs1)

    transformer_inputs2 = layers.Dense(448, activation="relu")(features2)
    transformer_inputs2 = layers.Reshape((7, 64), input_shape=transformer_inputs2.shape)(transformer_inputs2)

    cross_modal_transformer = CrossModalTransformer(d_model= 512, num_heads= 4, head_size=64, dff=64)
    cross_modal_transformer_out = cross_modal_transformer(transformer_inputs1, transformer_inputs2)

    cross_modal_transformer_out = layers.GlobalAveragePooling1D()(cross_modal_transformer_out)
    
    modality_out = layers.Dropout(0.30)(cross_modal_transformer_out)

    print("features_out shape", modality_out.shape)

    x = layers.Dense(512, activation="relu")(modality_out)
    
    x = layers.Dropout(0.30)(x)
    
    x = layers.Dense(128, activation="relu")(x)

    x = layers.Dropout(0.20)(x)
    
    x = layers.Dense(64, activation="relu")(x)

    x = layers.Dropout(0.10)(x)

    out = layers.Dense(5, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)

    model = keras.Model(inputs=[input_EEG, input_EMG, input_ECG, input_EOG, input_NEWAIR, input_THOR,
				input_ABDO, input_HC_EEG, input_HC_EMG, input_HC_ECG, input_HC_EOG,
				input_HC_NEWAIR, input_HC_THOR, input_HC_ABDO], outputs=out)
    
    return model

from sklearn.model_selection import ShuffleSplit

fold_var  = 1
cm_list = []
y_true_list = []
y_pred_list =[]
idx_list = list(np.arange(all_data_dict['EEG'].shape[0]))
#____________________________


from sklearn.model_selection import train_test_split
kf = StratifiedKFold(n_splits = 5, random_state = 22, shuffle = True)
import seaborn as sns
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.metrics import ConfusionMatrixDisplay

class ConfusionMatrixCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.conf_matrices = []

    def on_epoch_end(self, epoch, logs=None):
        # Obtain true labels and predicted labels for validation data
        y_true = np.argmax(self.validation_data[1], axis=1)
        print("y_true", y_true.shape)
        y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        #y_pred_prob = tf.nn.softmax(y_pred)
        #print("y_pred_prob", y_pred_prob.shape)
        
        print("Accuracy:", accuracy_score(y_true, y_pred))
        if accuracy_score(y_true, y_pred)>0.50:
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            print("Accuracy:", accuracy_score(y_true, y_pred))
            print("Cohen kappa:", cohen_kappa_score(y_true, y_pred))
            print("MF1:", f1_score(y_true, y_pred, average="macro"))
            print("Sensitivity:", sensitivity_score(y_true, y_pred, average='macro'))
            print("Specificity:", specificity_score(y_true, y_pred, average='macro'))
            print("Gmean: ", geometric_mean_score(y_true, y_pred, average=None).mean())
            print("Gmean: ", geometric_mean_score(y_true, y_pred, average=None))
            print("Gmean: ", geometric_mean_score(y_true, y_pred, average='micro'))
            print("Gmean: ", geometric_mean_score(y_true, y_pred, average='multiclass'))
            print("Gmean: ", geometric_mean_score(y_true, y_pred, average='weighted'))
            print("MCC: ", matthews_corrcoef(y_true, y_pred))
            #multi_class = 'ovo'
            #print("AUC score:", roc_auc_score(y_true, y_pred_prob, multi_class=multi_class))

            # Calculate confusion matrix
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true', cmap=plt.cm.Blues, display_labels=['Wake','N1','N2','N3','REM'])
            plt.show()

for train_idx, test_idx in kf.split(idx_list, all_data_dict['SleepStage']):
	print("\nCross-Validation, Fold:", fold_var,'\n')
	print("train num", len(train_idx))
	print("test num", len(test_idx))

	y_train, x_train_EEG, x_test_EEG, x_train_EMG, x_test_EMG, x_train_ECG, x_test_ECG, x_train_EOG, x_test_EOG, x_train_NEWAIR, x_test_NEWAIR,
	                                  x_train_THOR, x_test_THOR, x_train_ABDO, x_test_ABDO, spec_EEG1, spec_EEG2, spec_EMG, spec_ECG, spec_EOG1,
	                                  spec_EOG2, spec_NEWAIR, spec_THOR, spec_ABDO, spec_EEG1_test, spec_EEG2_test, spec_EMG_test, spec_ECG_test,
	                                  spec_EOG1_test, spec_EOG2_test, spec_NEWAIR_test, spec_THOR_test, spec_ABDO_test, EEG_HC, EEG_HC_test, EMG_HC,
	                                  EMG_HC_test, ECG_HC, ECG_HC_test, EOG_HC, EOG_HC_test, NEWAIR_HC, NEWAIR_HC_test, THOR_HC, THOR_HC_test, ABDO_HC,
	                                  ABDO_HC_test = split_and_normalize(all_data_dict, train_idx, test_idx)

	#y_train = all_data_dict['SleepStage'][train_idx]

	y_test = all_data_dict['SleepStage'][test_idx]
	print("Stages Train", Counter(y_train))
	print("Stages Test", Counter(y_test))
	#frames = [pd.DataFrame(y_train), pd.DataFrame(y_train), pd.DataFrame(y_train), pd.DataFrame(y_train)]
	#y_train = pd.concat(frames)
    
	y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=5)
	y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=5)
	
	print("x_train_EEG", x_train_EEG.shape)
	print("y_train_onehot", y_train_onehot.shape)

	class_w = class_weight.compute_class_weight('balanced', classes= np.unique(y_train), y = y_train)
	class_w_dict = dict(enumerate(class_w))

	print("x_test_EEG", x_test_EEG.shape)
	print("y_test_onehot", y_test_onehot.shape)
	#csv_logger = keras.callbacks.CSVLogger('training_log_CMST.csv', append=True)

	model = SleepStage()
	
	optimizer=keras.optimizers.Adam(learning_rate=1e-4)
					#, gradient_transformers=[AutoClipper(5)])
	
	model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
	
	conf_matrix_callback = ConfusionMatrixCallback(([x_test_EEG, x_test_EMG, x_test_ECG, x_test_EOG, x_test_NEWAIR, x_test_THOR, x_test_ABDO, EEG_HC_test,
							 EMG_HC_test, ECG_HC_test, EOG_HC_test, NEWAIR_HC_test, THOR_HC_test, ABDO_HC_test], y_test_onehot))

	
	hist = model.fit([x_train_EEG, x_train_EMG, x_train_ECG, x_train_EOG, x_train_NEWAIR, x_train_THOR, x_train_ABDO, EEG_HC, EMG_HC, ECG_HC,
			  EOG_HC, NEWAIR_HC, THOR_HC, ABDO_HC], y_train_onehot, epochs=400, shuffle=True, verbose=2, batch_size=512, validation_split=0.1,
			callbacks=[conf_matrix_callback] )
##	model_dir = "model/"
##	model.save(model_dir+'combine_3model_15sub'+str(fold_var)+'.h5')
	print(hist.history.keys())

	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='upper right')
	plt.show()

	plt.plot(hist.history['accuracy'])
	plt.plot(hist.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='lower right')
	plt.show()

	y_preds = model.predict([x_test_EEG, x_test_EMG, x_test_ECG, x_test_EOG, x_test_NEWAIR, x_test_THOR,
				 x_test_ABDO, EEG_HC_test, EMG_HC_test, ECG_HC_test, EOG_HC_test,
				 NEWAIR_HC_test, THOR_HC_test, ABDO_HC_test])
	
	y_preds = np.argmax(y_preds, axis=1)
	y_test = np.argmax(y_test_onehot, axis=1)
	print("Accuracy:", accuracy_score(y_test, y_preds))
	print(confusion_matrix(y_test, y_preds))
	print(classification_report(y_test, y_preds))
	print("Cohen kappa:", cohen_kappa_score(y_true, y_pred))
	print("MF1:", f1_score(y_true, y_pred, average="macro"))
	print("Sensitivity:", sensitivity_score(y_true, y_pred, average='macro'))
	print("Specificity:", specificity_score(y_true, y_pred, average='macro'))
	print("Gmean: ", geometric_mean_score(y_true, y_pred, average=None).mean())
	print("Gmean: ", geometric_mean_score(y_true, y_pred, average=None))
	print("Gmean: ", geometric_mean_score(y_true, y_pred, average='micro'))
	print("Gmean: ", geometric_mean_score(y_true, y_pred, average='multiclass'))
	print("Gmean: ", geometric_mean_score(y_true, y_pred, average='weighted'))
	print("MCC: ", matthews_corrcoef(y_true, y_pred))

	loss, acc = model.evaluate([x_test_EEG, x_test_EMG, x_test_ECG, x_test_EOG, x_test_NEWAIR, x_test_THOR,
				    x_test_ABDO, EEG_HC_test, EMG_HC_test, ECG_HC_test, EOG_HC_test, NEWAIR_HC_test,
				    THOR_HC_test, ABDO_HC_test], y_test_onehot)
	
	fold_var += 1
	
