import os
import re
import pyedflib
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter


edf_path = 'polysomnography/edfs/shhs/'
ann_path = 'polysomnography/annotations-events-nsrr/shhs/'

signal_info = sorted(['ABDO RES', 'ECG', 'EEG', 'EEG(sec)', 'EMG', 'EOG(L)', 'EOG(R)', 'H.R.', 'LIGHT', 'POSITION', 'SaO2', 'THOR RES']) # 'NEW AIR' by search
signal_info_with_air = sorted(['ABDO RES', 'ECG', 'EEG', 'EEG(sec)', 'EMG', 'EOG(L)', 'EOG(R)', 'H.R.', 'LIGHT', 'OX stat', 'POSITION', 'SaO2', 'THOR RES', 'NEW AIR'])
air_names = ['NEW AIR','NEWAIR','AIRFLOW','airflow','new A/F','New Air','AUX']
EEG2_names = ['EEG(sec)', 'EEG2']

sleep_stage_dict = {0.0:0, 1.0:1, 2.0:2, 3.0:3, 4.0:3, 5.0:4}
sig_samp_freq = {'ABDO RES':10,
				 'ECG':125,
				 'EEG':125,
				 'EEG(sec)':125,
				 'EMG':125,
				 'EOG(L)':50,
				 'EOG(R)':50,
				 'H.R.':1,
				 'LIGHT':1,
				 'NEW AIR':10,
				 'OX stat':1,
				 'POSITION':1,
				 'SaO2':1,
				 'THOR RES':10,
				 'SleepStage':1,
				 'Apnea':1}

def read_edfrecord(filename, FS = 125):
	signal_info = sorted(['ABDO RES', 'ECG', 'EEG', 'EMG', 'EOG(L)', 'EOG(R)', 'H.R.', 'LIGHT', 'POSITION', 'SaO2', 'THOR RES']) # 'EEG(sec)', AIRFLOW

	f = pyedflib.EdfReader(filename)
	print("Startdatetime: ", f.getStartdatetime())
			
	for air in air_names:
		if air in f.getSignalLabels():
			print("air flow name:", air)
			air_name = air
			signal_info.append(air_name)
			break

	for eeg2 in EEG2_names:
		if eeg2 in f.getSignalLabels():
			print("EEG2 name:", eeg2)
			eeg2_name = eeg2
			signal_info.append(eeg2)
			break
		
	EEG_chan = signal_info.index('EEG') # get the channel with highest FS, i.e. EEG
	N = f.getNSamples()[EEG_chan]
	time_index = np.arange(f.file_duration,step=1/FS)

	all_df = pd.DataFrame(columns = signal_info_with_air, index=time_index.round(decimals=3)) # round(decimals=3)is to help match when assigning sub_df to df

	signal_label_info = f.getSignalLabels()
	for s in signal_info:
		if s not in signal_label_info:
			print(s,"not in signal_label_info:", signal_label_info)

	print(f.getSignalLabels())
	print(f.getSampleFrequencies())
	for s in signal_info:        
		s_chan = signal_label_info.index(s)
		sigbuf = f.readSignal(s_chan)
		sig_fs = f.getSampleFrequency(s_chan)
		sig_tm = np.arange(f.file_duration,step=1/sig_fs)

		
		if sig_fs != FS and 'EEG' not in s: #and 'EEG' not in s and 'H.R.' not in s
			for i in range(len(sig_tm)):
				if sig_tm[i] %30 != 0:
					sig_tm[i] = sig_tm[i]//(1/125)/125
		else:
			sig_tm = sig_tm.round(decimals=3)
		assert len(sigbuf) == len(sig_tm), "signal and its time are not match"

		sub_df = pd.DataFrame({s:sigbuf}, index=sig_tm)
		if s == air_name:
			all_df['NEW AIR'] = sub_df
		if s == eeg2_name:
			all_df['EEG(sec)'] = sub_df
		else:
			all_df[s] = sub_df

	f.close()


	return all_df

	
def read_annot_stage_regex(filename, df, FS=125):

	df['SleepStage'] = np.nan
	
	with open(filename, 'r') as f:
		content = f.read()
	

	patterns_start = re.findall(
		r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', 
		content)
	print("patterns_start ",patterns_start )
	assert len(patterns_start) == 1
	
	patterns_stages = re.findall(
		r'<EventType>Stages.Stages</EventType>\n' +
		r'<EventConcept>.+</EventConcept>\n' +
		r'<Start>[0-9\.]+</Start>\n' +
		r'<Duration>[0-9\.]+</Duration>', 
		content)
	

	stages = []
	starts = []
	durations = []
	for pattern in patterns_stages:
		lines = pattern.splitlines()
		stageline = lines[1]
		stage = int(stageline[-16])
		startline = lines[2]
		start = float(startline[7:-8])
		durationline = lines[3]
		duration = float(durationline[10:-11])
		assert duration % 30 == 0.

		df.loc[start:(start+duration-1/FS),'SleepStage']=stage

		epochs_duration = int(duration) // 30
		stages += [stage]*epochs_duration
		starts += [start]
		durations += [duration]

	assert int((start + duration)/30) == len(stages)
	print("len(stages)",len(stages))
	return df


def preprocess(nsrrid_list, data_dir, output_dir):

	for s in nsrrid_list:
		print('preparing',s)
		filename = data_dir+edf_path+s+'.edf'
		df = read_edfrecord(filename)
		
		

		filename = data_dir+ann_path+s+'-nsrr.xml'
		df = read_annot_stage_regex(filename,df=df)
		
		

		df.to_pickle(output_dir+s+'.pkl')

	print('preprocess done')   


def df2dict(df):
	############################# Segmentation #################################
	sig_names = df.columns.to_list()
	data_dict = {k : [] for k in sig_names}
	#print(data_dict['SleepStage'])

	FS = 125
	epoch_size = 30*FS
	
	for s in sig_names:
		start=0
		for i in range(0, len(df), epoch_size):
				sig_epoch = df[s].iloc[start:start+epoch_size].dropna().to_numpy()
				data_dict[s].append(sig_epoch)  
				start = start+epoch_size

	# decide sleep stage label
	for i in range(len(data_dict['SleepStage'])):
		if np.all(data_dict['SleepStage'][i] == data_dict['SleepStage'][i][0]):

			# data_dict['SleepStage'][i] = data_dict['SleepStage'][i][0] if data_dict['SleepStage'][i][0] != 5 else 4 # change REM stage 5 to 4
			data_dict['SleepStage'][i] = sleep_stage_dict[data_dict['SleepStage'][i][0]]
			#print("data_dict['SleepStage'][i]",data_dict['SleepStage'][i])
		else:
			print("error: sleep stage labels in one epoch are not equal")
	

	############################# Remove before and after Wake #################################
	sleep_start = 0
	sleep_end = len(data_dict['SleepStage'])
	for i in range(len(data_dict['SleepStage'])):
		if data_dict['SleepStage'][i] != 0:
			sleep_start = i
			break
	for i in range(len(data_dict['SleepStage'])-1,-1,-1):
		if data_dict['SleepStage'][i] != 0:
			sleep_end = i
			break

	keep_wake_epochs = 15*2     # 15 min 
	if sleep_start > keep_wake_epochs:
		keep_start_idx = sleep_start - keep_wake_epochs
	else:
		keep_start_idx = 0
	
	if (len(data_dict['SleepStage'])-sleep_end) > keep_wake_epochs:
		keep_end_idx = sleep_end + keep_wake_epochs
	else:
		keep_end_idx = len(data_dict['SleepStage'])
						   
	for s in sig_names:
		data_dict[s] = np.array(data_dict[s][keep_start_idx:keep_end_idx-1]) 

	
	print("SleepStage", Counter(data_dict['SleepStage']))
	print("")
	
						   
	return data_dict

def segment(segmented_dir, processed_data_dir):
	fnames = sorted(os.listdir(processed_data_dir))
	for i in range(len(fnames[:])):
		#print(fnames[i])
		df = pd.read_pickle(processed_data_dir+fnames[i])
		data_dict = df2dict(df)

		with open(segmented_dir+fnames[i], 'wb') as handle:
			pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("segment done")

if __name__ == "__main__":
	data_dir = 'shhs/'
	processed_data_dir = 'data/processed_data/'
	data_dire = 'D:\sleepStaging\shhs\polysomnography\edfs\shhs'
	nsrrid_list_id = []
	names = sorted(os.listdir(data_dire))
	print(names)
	for i in range(len(names)):
		string = names[i]
		names[i] = string[6:12]
		nsrrid_list_id.append(names[i])
		
	print('nsrrid_list_id',nsrrid_list_id)

	#nsrrid_list_id = np.arange(200001,200051)
	#nsrrid_list_id = np.load('data/nsrrid_list_id.npy')
	#@nsrrid_list = ['shhs1-'+str(x) for x in nsrrid_list_id]
	#preprocess(nsrrid_list = nsrrid_list[:], data_dir=data_dir, output_dir=processed_data_dir)
	
	segmented_dir = 'data/segmented_data/'
	segment(segmented_dir = segmented_dir, processed_data_dir = processed_data_dir)

