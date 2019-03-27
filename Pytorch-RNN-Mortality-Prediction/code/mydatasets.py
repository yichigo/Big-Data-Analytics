import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.

	df = pd.read_csv(path)
	labels = df['y'].values
	labels = labels - 1
	data0 = df.loc[:, 'X1':'X178'].values

	if model_type == 'MLP':
		data = torch.from_numpy(data0.astype('float32'))
		target = torch.from_numpy(labels.astype('long'))#.view(-1,1)
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = torch.from_numpy(data0.astype('float32')).unsqueeze(1)
		target = torch.from_numpy(labels.astype('long'))
		dataset = TensorDataset(data, target)
	elif model_type == 'RNN':
		data = torch.from_numpy(data0.astype('float32')).unsqueeze(2)
		target = torch.from_numpy(labels.astype('long'))
		dataset = TensorDataset(data, target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# TODO: Calculate the number of features (diagnoses codes in the train set)
	features = set()
	for seq_patient in seqs:
		for seq_visit in seq_patient:
			for feature in seq_visit:
				features.add(feature)
	num_features = len(features)
	return num_features


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		self.seqs = []
		for seq_patient in seqs: # create a Sparse matrix for each patient
			row = []
			col = []
			for i, seq_visit in enumerate(seq_patient):
				row.extend([i] * len(seq_visit))
				col.extend(seq_visit)
			data = [1] * len(row)

			sparseMatrix = sparse.coo_matrix((data, (row, col)), shape=(len(seq_patient), num_features) )
			self.seqs.append(sparseMatrix)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence
	
	# sorted by the length of visits in descending order
	batch.sort(key=lambda x: np.shape(x[0].toarray())[0], reverse=True)
	seqs, labels = zip(*batch) # unzip

	batch_size = len(seqs)
	max_length = np.shape(seqs[0].toarray())[0]
	num_features = np.shape(seqs[0].toarray())[1]

	matrice = np.zeros((batch_size, max_length, num_features))
	lengths = []
	for i, seq_patient in enumerate(seqs):
		matrix = seq_patient.toarray()
		lengths.append(np.shape(matrix)[0]) # num of row = visit times for a patient
		matrice[i,:lengths[i],:]= matrix

	seqs_tensor = torch.FloatTensor(matrice)
	lengths_tensor = torch.LongTensor(lengths)
	labels_tensor = torch.LongTensor(labels) # Longint for labels

	return (seqs_tensor, lengths_tensor), labels_tensor
