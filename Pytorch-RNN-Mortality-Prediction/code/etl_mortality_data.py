import os
import pickle
import pandas as pd
import numpy as np

PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object)
	# Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# Read the homework description carefully.
	if icd9_str[0] != 'E':
		converted = icd9_str[:3]
	else:
		converted = icd9_str[:4]

	return converted


def build_codemap(df_icd9, transform):
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	# We build a code map using ONLY train data. Think about how to construct validation/test sets using this.
	
	df_digits = df_icd9['ICD9_CODE'].apply(transform)
	unique_digits = df_digits.unique()
	unique_id = np.arange(len(unique_digits))
	codemap = dict(zip(unique_digits, unique_id))
	return codemap


def create_dataset(path, codemap, transform):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:param transform: e.g. convert_icd9
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# 1. Load data from the three csv files
	# Loading the mortality file is shown as an example below. Load two other files also.
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
	df_admissions = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))
	df_diagnoses = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))

	# 2. Convert diagnosis code in to unique feature ID.
	# use 'transform(convert_icd9)' you implemented and 'codemap'.
	df_diagnoses['ICD9_CODE'] = df_diagnoses['ICD9_CODE'].apply(transform)
	df_diagnoses['FEATURE_ID'] = df_diagnoses['ICD9_CODE'].map(codemap)

	# drop the FEATURE_ID that not in the training dataset
	df_diagnoses = df_diagnoses[pd.notnull(df_diagnoses['FEATURE_ID'])]
	df_diagnoses['FEATURE_ID'] = df_diagnoses['FEATURE_ID'].astype(int)

	# 3. Group the diagnosis codes for the same visit.
	df_diagnoses = pd.DataFrame(df_diagnoses.groupby(['HADM_ID'])['FEATURE_ID'].apply(list))

	# 4. Group the visits for the same patient.
	df_admissions = df_admissions[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]
	df_admissions = df_admissions.sort_values(['SUBJECT_ID','ADMITTIME'],ascending=True)

	df = df_admissions.join(df_diagnoses, on = 'HADM_ID')
	df = pd.DataFrame(df.groupby(['SUBJECT_ID'])['FEATURE_ID'].apply(list))
	df = df.join(df_mortality.set_index('SUBJECT_ID'), on = 'SUBJECT_ID')

	# 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
	seq_data = list(df['FEATURE_ID'].values)

	# Visits for each patient must be sorted in chronological order.  (it has been sorted in step 4)
	# 6. Make patient-id List and label List also.
	patient_ids = list(df.index.values)
	labels = list(df['MORTALITY'].values)

	# The order of patients in the three List output must be consistent. (be consistent by join function)
	return patient_ids, labels, seq_data


def main():
	# Build a code map from the train set
	print("Build feature id map")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
