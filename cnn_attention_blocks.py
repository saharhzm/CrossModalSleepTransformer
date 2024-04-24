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

def extractor_10(x, fs, name='data'):
    
    print("extractor for sampling rate 10")
    
    inputs = x    
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(3, 3), padding="valid",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5),
			  kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(3, 3), padding="valid",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x1 = layers.Dropout(0.50)(x)
#_______________________________

    x = layers.Conv1D(filters=64, 
			  kernel_size=max(3, 3), padding="valid",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x1)
    x = layers.Conv1D(filters=128, 
			  kernel_size=max(3, 3), padding="valid",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x = layers.Dropout(0.50)(x)

    x = layers.Conv1D(filters=64, 
			  kernel_size=max(2, 2), padding="valid",
			  strides=max(1, 1), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(2, 2), padding="valid",
			  strides=max(1, 1), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x2 = layers.Dropout(0.50)(x)
    x = layers.Concatenate(axis=1)([x1, x2])
    x = layers.Dropout(0.50)(x)
    x = layers.GaussianNoise(0.1)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)
    print("x shape", x.shape)
    att_out = attention()(x)
    x = layers.Concatenate(axis=1)([cnn_out, att_out])
    print("x", x.shape)
    return x


def extractor_50(x, fs, name='data'):
    
    print("extractor  for sampling rate 50 EOGs")
    
    inputs = x    
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(8, 6), padding="valid",
			  strides=max(3, 3), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5),
			  kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(6, 6), padding="valid",
			  strides=max(3, 3), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x1 = layers.Dropout(0.50)(x)
#_______________________________

    x = layers.Conv1D(filters=64, 
			  kernel_size=max(5, 5), padding="valid",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x1)
    x = layers.Conv1D(filters=128, 
			  kernel_size=max(5, 5), padding="valid",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x = layers.Dropout(0.50)(x)

    x = layers.Conv1D(filters=64, 
			  kernel_size=max(3, 3), padding="valid",
			  strides=max(1, 1), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(2, 2), padding="valid",
			  strides=max(1, 1), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x2 = layers.Dropout(0.50)(x)
    x = layers.Concatenate(axis=1)([x1, x2])
    x = layers.Dropout(0.50)(x)
    x = layers.GaussianNoise(0.1)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)
    print("x shape", x.shape)
    att_out = attention()(x)
    x = layers.Concatenate(axis=1)([cnn_out, att_out])
    print("x", x.shape)
    return x


def extractor_125(x, fs, name='data'):
    
    print("extractor  for sampling rate 125 EEGs, EMG, ECG")
    
    inputs = x    
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(15, 15), padding="valid",
			  strides=max(5, 5), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5),
			  kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(8, 8), padding="valid",
			  strides=max(3, 3), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x1 = layers.Dropout(0.50)(x)
#_______________________________

    x = layers.Conv1D(filters=64, 
			  kernel_size=max(5, 5), padding="valid",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x1)
    x = layers.Conv1D(filters=64, 
			  kernel_size=max(5, 5), padding="valid",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x = layers.Dropout(0.50)(x)

    x = layers.Conv1D(filters=128, 
			  kernel_size=max(3, 3), padding="valid",
			  strides=max(1, 1), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.Conv1D(filters=256, 
			  kernel_size=max(3, 3), padding="valid",
			  strides=max(1, 1), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x = layers.Dropout(0.50)(x)

    x = layers.Conv1D(filters=128, 
			  kernel_size=max(2, 2), padding="valid",
			  strides=max(1, 1), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(2, 2), padding="valid",
			  strides=max(1, 1), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x2 = layers.Dropout(0.50)(x)
    x = layers.Concatenate(axis=1)([x1, x2])
    x = layers.Dropout(0.50)(x)
    x = layers.GaussianNoise(0.1)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)
    print("x shape", x.shape)
    att_out = attention()(x)
    x = layers.Concatenate(axis=1)([cnn_out, att_out])
    print("x", x.shape)
    return x


def extractor_spec(x, fs, name='data'):
    
    print("extractor  for spectural features")

    
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(3, 3), padding="same",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5),
			  kernel_initializer=tf.keras.initializers.HeNormal())(x)
    
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(3, 3), padding="same",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x1 = layers.Dropout(0.50)(x)
    print("x1 shape", x1.shape)

    x = layers.Conv1D(filters=64, 
			  kernel_size=max(3, 3), padding="same",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x1)
        
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(3, 3), padding="same",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
            
    x = layers.MaxPool1D((2), strides=2)(x)
    x2 = layers.Dropout(0.50)(x)
    x = layers.Concatenate(axis=1)([x1, x2])
    x = layers.Dropout(0.50)(x)
    x = layers.GaussianNoise(0.1)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)
    print("x shape", x.shape)
    att_out = attention()(x)
    x = layers.Concatenate(axis=1)([cnn_out, att_out])
    return x


def extractor_HC(x, fs, name='data'):

    print("extractor  for handcrafted features)
    
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(3, 3), padding="same",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5),
			  kernel_initializer=tf.keras.initializers.HeNormal())(x)
    
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(3, 3), padding="same",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
    x = layers.MaxPool1D((2), strides=2)(x)
    x1 = layers.Dropout(0.50)(x)
    print("x1 shape", x1.shape)

    x = layers.Conv1D(filters=64, 
			  kernel_size=max(3, 3), padding="same",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x1)
        
    x = layers.Conv1D(filters=32, 
			  kernel_size=max(3, 3), padding="same",
			  strides=max(2, 2), activation='relu',
			  activity_regularizer=tf.keras.regularizers.L1(1e-5))(x)
            
    x = layers.MaxPool1D((2), strides=2)(x)
    x2 = layers.Dropout(0.50)(x)
    x = layers.Concatenate(axis=1)([x1, x2])
    x = layers.Dropout(0.50)(x)
    x = layers.GaussianNoise(0.1)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)
    print("x shape", x.shape)
    att_out = attention()(x)
    x = layers.Concatenate(axis=1)([cnn_out, att_out])
    return x


######### Additive attention mechanism##########
class attention(Layer):

    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
