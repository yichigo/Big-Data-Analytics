import pytest
from delayed_assert import expect, assert_expectations
import io
import os
import pickle

import scipy as sp
import numpy as np
import pandas as pd
import torch

from mydatasets import load_seizure_dataset, calculate_num_features, VisitSequenceWithLabelDataset, visit_collate_fn
from etl_mortality_data import convert_icd9, build_codemap, create_dataset
from mymodels import *

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')

example_seizure_file = "X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,X30,X31,X32,X33,X34,X35,X36,X37,X38,X39,X40,X41,X42,X43,X44,X45,X46,X47,X48,X49,X50,X51,X52,X53,X54,X55,X56,X57,X58,X59,X60,X61,X62,X63,X64,X65,X66,X67,X68,X69,X70,X71,X72,X73,X74,X75,X76,X77,X78,X79,X80,X81,X82,X83,X84,X85,X86,X87,X88,X89,X90,X91,X92,X93,X94,X95,X96,X97,X98,X99,X100,X101,X102,X103,X104,X105,X106,X107,X108,X109,X110,X111,X112,X113,X114,X115,X116,X117,X118,X119,X120,X121,X122,X123,X124,X125,X126,X127,X128,X129,X130,X131,X132,X133,X134,X135,X136,X137,X138,X139,X140,X141,X142,X143,X144,X145,X146,X147,X148,X149,X150,X151,X152,X153,X154,X155,X156,X157,X158,X159,X160,X161,X162,X163,X164,X165,X166,X167,X168,X169,X170,X171,X172,X173,X174,X175,X176,X177,X178,y\n" \
					   "-104,-96,-96,-99,-82,-100,-98,-107,-97,-87,-67,-75,-77,-80,-76,-80,-76,-54,-52,-51,-59,-75,-90,-91,-98,-84,-74,-81,-69,-76,-70,-69,-82,-77,-77,-68,-67,-70,-68,-78,-66,-64,-67,-59,-62,-76,-83,-76,-85,-82,-65,-68,-87,-94,-87,-96,-80,-96,-89,-79,-58,-39,-38,-38,-45,-64,-64,-61,-72,-74,-61,-60,-64,-59,-49,-39,-36,-26,-43,-52,-63,-57,-74,-73,-72,-53,-65,-46,-47,-63,-51,-71,-54,-63,-52,-54,-69,-65,-61,-63,-60,-70,-61,-82,-84,-92,-91,-85,-74,-78,-54,-53,-60,-65,-70,-61,-47,-55,-69,-71,-64,-81,-82,-82,-100,-89,-81,-82,-75,-69,-63,-79,-97,-94,-98,-72,-50,-60,-57,-66,-74,-73,-76,-56,-56,-53,-44,-58,-59,-67,-88,-86,-87,-86,-86,-77,-62,-32,-16,-22,-37,-55,-75,-72,-77,-68,-75,-75,-76,-83,-77,-88,-76,-86,-69,-77,-74,-73,5\n" \
					   "-58,-16,56,119,134,133,97,74,42,-9,-43,-80,-105,-135,-113,-112,-63,-37,4,35,67,73,38,28,32,26,27,40,9,-28,-56,-79,-91,-76,-65,-40,-31,5,52,89,113,108,98,59,25,-19,-41,-70,-49,-9,18,44,60,69,66,73,63,46,31,29,21,-11,-15,-41,-21,3,64,111,139,133,104,64,25,11,-7,-3,6,5,3,-4,-16,-22,3,22,50,71,64,69,82,83,70,64,53,51,19,-12,-30,-48,-49,-43,-36,-29,-18,15,41,72,87,90,61,35,26,7,-5,-13,-43,-49,-63,-75,-60,-30,-2,63,93,115,115,105,60,6,-22,-50,-42,-37,-40,-24,-9,16,38,40,73,97,110,112,89,53,1,-33,-39,-45,-60,-64,-66,-56,-51,-25,4,32,42,76,98,126,138,124,101,48,-30,-84,-114,-135,-141,-126,-107,-82,-70,-50,-36,8,66,123,4\n" \
					   "-287,-256,-222,-185,-178,-166,-165,-138,-120,-83,-55,-42,-11,22,42,69,62,82,72,61,17,-42,-124,-180,-218,-242,-251,-258,-233,-209,-168,-116,-79,-51,-7,6,43,49,61,78,91,107,106,94,77,39,5,-86,-138,-141,-144,-127,-152,-168,-183,-221,-235,-240,-228,-203,-163,-93,-29,37,76,89,99,119,114,131,116,118,108,126,109,123,116,94,56,30,-35,-93,-93,-79,-40,-46,-117,-179,-203,-210,-211,-177,-154,-105,-59,-10,-6,11,30,38,44,53,83,108,129,141,153,166,160,152,157,148,136,136,117,94,33,-67,-196,-318,-358,-310,-252,-200,-193,-178,-168,-150,-118,-96,-42,-1,21,39,47,47,43,62,63,77,80,87,85,62,57,52,63,57,69,71,53,16,-12,-35,-23,2,40,82,132,172,248,356,413,426,281,70,-151,-313,-391,-407,-370,-346,-284,-231,-173,-135,-80,-39,1\n"


def test_load_seizure_dataset_mlp():
	example = io.StringIO(example_seizure_file)
	tensor_dataset = load_seizure_dataset(example, 'MLP')

	expect(type(tensor_dataset) == torch.utils.data.dataset.TensorDataset, "it should return TensorDataset")

	data_tensor, target_tensor = tensor_dataset.tensors

	expect(data_tensor.size() == (3, 178), "it does not have expected shapes")
	expect(target_tensor.size() == (3,), "it does not have expected shapes")
	assert_expectations()


def test_load_seizure_dataset_cnn():
	example = io.StringIO(example_seizure_file)
	tensor_dataset = load_seizure_dataset(example, 'CNN')

	expect(type(tensor_dataset) == torch.utils.data.dataset.TensorDataset, "it should return TensorDataset")

	data_tensor, target_tensor = tensor_dataset.tensors

	expect(data_tensor.size() == (3, 1, 178), "it does not have expected shapes")
	expect(target_tensor.size() == (3,), "it does not have expected shapes")
	assert_expectations()


def test_load_seizure_dataset_rnn():
	example = io.StringIO(example_seizure_file)
	tensor_dataset = load_seizure_dataset(example, 'RNN')

	expect(type(tensor_dataset) == torch.utils.data.dataset.TensorDataset, "it should return TensorDataset")

	data_tensor, target_tensor = tensor_dataset.tensors

	expect(data_tensor.size() == (3, 178, 1), "it does not have expected shapes")
	expect(target_tensor.size() == (3,), "it does not have expected shapes")
	assert_expectations()


def model_eval(model, data_loader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	with torch.no_grad():
		data_iter = iter(data_loader)
		input, target = data_iter.next()

		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)

		try:
			_ = model(input)
			return None
		except BaseException as err:
			return err


def eval_seizure_model(model, model_type):
	example = io.StringIO(example_seizure_file)
	dataset = load_seizure_dataset(example, model_type)
	loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)

	return model_eval(model, loader)


def test_saved_model_mlp():
	model = torch.load(os.path.join(project_root, "output/seizure/MyMLP.pth"), map_location=lambda storage, loc: storage)
	any_exception = eval_seizure_model(model, 'MLP')
	assert not any_exception, "your saved model should be matched with your model definition"


def test_saved_model_cnn():
	model = torch.load(os.path.join(project_root, "output/seizure/MyCNN.pth"), map_location=lambda storage, loc: storage)
	any_exception = eval_seizure_model(model, 'CNN')
	assert not any_exception, "your saved model should be matched with your model definition"


def test_saved_model_rnn():
	model = torch.load(os.path.join(project_root, "output/seizure/MyRNN.pth"), map_location=lambda storage, loc: storage)
	any_exception = eval_seizure_model(model, 'RNN')
	assert not any_exception, "your saved model should be matched with your model definition"


def test_convert_icd9():
	example = pd.DataFrame(['V1046', 'V090', '135', '1890', '19889', 'E9352', 'E935'], columns=['ICD9_CODE'])
	expected = pd.Series(['V10', 'V09', '135', '189', '198', 'E935', 'E935'])
	actual = example['ICD9_CODE'].apply(convert_icd9)

	assert actual.equals(expected)


def test_build_codemap():
	example = pd.DataFrame(['V1046', 'V090', '135', '1890', '19889', 'E9352', 'E935'], columns=['ICD9_CODE'])
	actual = build_codemap(example, convert_icd9)
	expect(isinstance(actual, dict), "it should be a dict object")
	expect(len(actual) == 6)
	expect(len(actual) == max(actual.values())+1, "the range of ids should be [0, N-1]")
	assert_expectations()


def test_create_dataset():
	PATH_TRAIN = os.path.join(project_root, "data/mortality/train")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"),
						  usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	ids, labels, seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	expect(isinstance(ids, list), "it should be a list object")
	expect(isinstance(labels, list), "it should be a list object")
	expect(isinstance(seqs, list), "it should be a list object")
	expect(isinstance(seqs[0], list), "seq data should be a list of list(s) of list(s)")
	expect(isinstance(seqs[0][0], list), "seq data should be a list of list(s) of list(s)")

	expect(len(ids) == 5273, "len(ids) is not matched")
	expect(len(labels) == 5273, "len(labels) is not matched")
	expect(len(seqs) == 5273, "len(seqs) is not matched")

	assert_expectations()


def test_calculate_num_features():
	# using the validation data only for the test purpose
	# You should use the train set in your code.
	PATH_VALID_SEQS = os.path.join(project_root, "data/mortality/processed/mortality.seqs.validation")
	valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
	num_features = calculate_num_features(valid_seqs)

	assert num_features == 902


def test_visit_sequence_with_label_dataset():
	PATH_VALID_SEQS = os.path.join(project_root, "data/mortality/processed/mortality.seqs.validation")
	PATH_VALID_LABELS = os.path.join(project_root, "data/mortality/processed/mortality.labels.validation")
	valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
	valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
	num_features = calculate_num_features(valid_seqs)
	dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features)
	expect(isinstance(dataset.seqs, list), "dataset.seqs should be a list")
	expect(isinstance(dataset.labels, list), "dataset.labels should be a list")
	expect(len(valid_seqs) == len(dataset.seqs), "length of the dataset is not matched")

	sample = dataset.seqs[0]

	expect(isinstance(sample, np.ndarray) or isinstance(sample.toarray(), np.ndarray), "each seqs element should be a numpy array or scipy.sparse matrix")

	expect(sample.shape[0] == len(valid_seqs[0]), "number of rows for the first patient not matched")
	expect(sample.shape[1] == num_features, "number of cols and the number of features are not matched with each other")

	assert_expectations()


def test_visit_collate_fn():
	PATH_VALID_SEQS = os.path.join(project_root, "data/mortality/processed/mortality.seqs.validation")
	PATH_VALID_LABELS = os.path.join(project_root, "data/mortality/processed/mortality.labels.validation")
	valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
	valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
	num_features = calculate_num_features(valid_seqs)
	dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features)

	sample_batch = list(zip(dataset.seqs, dataset.labels))[:3]
	(seqs_tensor, lengths_tensor), labels_tensor = visit_collate_fn(sample_batch)

	expect(isinstance(seqs_tensor, torch.Tensor), "it should be Tensor")
	expect(isinstance(lengths_tensor, torch.Tensor), "it should be Tensor")
	expect(isinstance(labels_tensor, torch.Tensor), "it should be Tensor")
	expect(seqs_tensor.dtype == torch.float32, "seqs should be FloatTensor")
	expect(lengths_tensor.dtype == torch.int64, "lengths should be FloatTensor")
	expect(labels_tensor.dtype == torch.int64, "labels should be FloatTensor")

	assert_expectations()


def test_saved_model_variable_rnn():
	PATH_VALID_SEQS = os.path.join(project_root, "data/mortality/processed/mortality.seqs.validation")
	PATH_VALID_LABELS = os.path.join(project_root, "data/mortality/processed/mortality.labels.validation")
	valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
	valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
	num_features = calculate_num_features(valid_seqs)
	dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features)
	loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=3, shuffle=False, collate_fn=visit_collate_fn, num_workers=0)

	model = torch.load(os.path.join(project_root, "output/mortality/MyVariableRNN.pth"), map_location=lambda storage, loc: storage)

	any_exception = model_eval(model, loader)

	assert not any_exception, "your saved model should be matched with your model definition"
