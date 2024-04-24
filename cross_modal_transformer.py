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
from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder


class CrossModalityAttention(layers.Layer):
    def __init__(self, num_heads, d_model):
        super(CrossModalityAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_size = d_model // num_heads

        self.W_q_raw_data = layers.Dense(d_model)
        self.W_k_raw_data = layers.Dense(d_model)
        self.W_v_raw_data = layers.Dense(d_model)

        self.W_q_handcraft_data = layers.Dense(d_model)
        self.W_k_handcraft_data = layers.Dense(d_model)
        self.W_v_handcraft_data = layers.Dense(d_model)

        self.output_dense = layers.Dense(d_model)
        
    @tf.function
    def call(self, raw_data_inputs, handcraft_data_inputs):
        
        #batch_size, raw_data_length, _ = tf.shape(raw_data_inputs)
        batch_size = tf.shape(raw_data_inputs)[0]
        raw_data_length = tf.shape(raw_data_inputs)[1]
        
        #batch_size, handcraft_data_length, _ = tf.shape(handcraft_data_inputs)
        batch_size = tf.shape(handcraft_data_inputs)[0]
        handcraft_data_length = tf.shape(handcraft_data_inputs)[1]
        
        # raw_data modality attention
        Q_raw_data = self.W_q_raw_data(raw_data_inputs)
        K_raw_data = self.W_k_raw_data(raw_data_inputs)
        V_raw_data = self.W_v_raw_data(raw_data_inputs)

        # handcraft_data modality attention
        Q_handcraft_data = self.W_q_handcraft_data(handcraft_data_inputs)
        K_handcraft_data = self.W_k_handcraft_data(handcraft_data_inputs)
        V_handcraft_data = self.W_v_handcraft_data(handcraft_data_inputs)

        # Compute attention weights and context vectors
        attention_weights_raw_data = tf.nn.softmax(tf.matmul(Q_raw_data, K_handcraft_data, transpose_b=True))
        context_raw_data = tf.matmul(attention_weights_raw_data, V_raw_data)

        attention_weights_handcraft_data = tf.nn.softmax(tf.matmul(Q_handcraft_data, K_raw_data, transpose_b=True))
        context_handcraft_data = tf.matmul(attention_weights_handcraft_data, V_handcraft_data)

        # Combine modalities using context vectors
        #combined_context = context_raw_data + context_handcraft_data

        # Apply a dense layer to the combined context
        #output = self.output_dense(combined_context)

        return context_raw_data, context_handcraft_data

#_________cross modal transformer____________
    
class CrossModalTransformer(keras.Model):
    def __init__(self, d_model, num_heads, head_size, dff):
        super(CrossModalTransformer, self).__init__()

        # raw_data and handcraft_data encoders
        
        self.raw_data_encoder =  TransformerEncoder(intermediate_dim=head_size, num_heads=num_heads)
        #raw_dataEncoder(num_layers, d_model, num_heads, dff, input_vocab_size)
        
        self.handcraft_data_encoder = TransformerEncoder(intermediate_dim=head_size, num_heads=num_heads)
        #handcraft_dataEncoder(num_layers, d_model, num_heads, dff, input_vocab_size)
        
        # Cross-Modality Attention Layer
        self.cross_modality_attention = CrossModalityAttention(num_heads, d_model)

        # Decoder
        self.decoder = TransformerDecoder(intermediate_dim=head_size, num_heads=num_heads)
        #Decoder(num_layers, d_model, num_heads, dff, target_vocab_size)
        
    def call(self, raw_data_input, handcraft_data_input):
        
        # Encode raw_data and handcraft_data modalities
        raw_data_encoded = self.raw_data_encoder(inputs = raw_data_input)
        
        handcraft_data_encoded = self.handcraft_data_encoder(inputs = handcraft_data_input)

        # Apply cross-modality attention
        context_raw_data, context_handcraft_data = self.cross_modality_attention(raw_data_encoded, handcraft_data_encoded)
        
        # Combine modalities using context vectors
        # Encoder output
        encoder_output = tf.multiply([context_raw_data, context_handcraft_data], axis=-1)

        # Decoder input (concatenate or any other method based on your architecture)
        decoder_input = tf.concat([raw_data_input, handcraft_data_input], axis=-1)

        # Pass through the decoder
        output = self.decoder(decoder_sequence = decoder_input, encoder_sequence = encoder_output)

        return output





    
