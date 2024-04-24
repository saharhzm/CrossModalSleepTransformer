import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter

def split_and_normalize(all_data_dict, train_idx, test_idx):
        
	x1 = all_data_dict['EEG']
	scaler = StandardScaler()
	x1 = scaler.fit_transform(x1)
	x2 = all_data_dict['EEG(sec)']
	scaler = StandardScaler()
	x2 = scaler.fit_transform(x2)
	x = np.stack([x1,x2], axis=2)
	x_train_EEG = x[train_idx]
	x_test_EEG = x[test_idx]

	x = all_data_dict['EMG']
	scaler = StandardScaler()
	x = scaler.fit_transform(x)

	x_train_EMG = x[train_idx]
	x_test_EMG = x[test_idx]

	x = all_data_dict['ECG']
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	x_train_ECG = x[train_idx]
	x_test_ECG = x[test_idx]

	x1 = all_data_dict['EOG(L)']
	scaler = StandardScaler()
	x1 = scaler.fit_transform(x1)
	x2 = all_data_dict['EOG(R)']
	scaler = StandardScaler()
	x2 = scaler.fit_transform(x2)
	x = np.stack([x1,x2], axis=2)
	x_train_EOG = x[train_idx]
	x_test_EOG = x[test_idx]

	x = all_data_dict['NEW AIR']
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	x_train_NEWAIR = x[train_idx]
	x_test_NEWAIR = x[test_idx]

	x = all_data_dict['THOR RES']
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	x_train_THOR = x[train_idx]
	x_test_THOR = x[test_idx]


	x = all_data_dict['ABDO RES']
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	x_train_ABDO = x[train_idx]
	x_test_ABDO = x[test_idx]

	return x_train_EEG, x_test_EEG, x_train_EMG, x_test_EMG, x_train_ECG, x_test_ECG, x_train_EOG, x_test_EOG, x_train_NEWAIR, x_test_NEWAIR, x_train_THOR, x_test_THOR, x_train_ABDO, x_test_ABDO

def lr_schedule(epoch, lr):    
        if epoch == 400 :
            lr = 1e-4
        return lr


def load_data(dict_data_dir):
	fnames = sorted(os.listdir(dict_data_dir))
	all_data_dict = dict()
	for i in range(len(fnames)):
		print(fnames[i])
		data_dict = pd.read_pickle(dict_data_dir+fnames[i])
		#print("Stages",fnames[i], Counter(data_dict['SleepStage']))

		for k in data_dict:
			if k in all_data_dict:
				all_data_dict[k] = np.concatenate((all_data_dict[k],data_dict[k]),axis=0)
			else:
				all_data_dict[k] = data_dict[k]
	return all_data_dict
