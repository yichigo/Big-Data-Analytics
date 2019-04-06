import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		# baseline
		# self.hidden1 = nn.Linear(178, 16)
		# self.out = nn.Linear(16, 5)
		# improved
		self.hidden1 = nn.Linear(178, 256)
		self.hidden2 = nn.Linear(256, 256)
		self.hidden3 = nn.Linear(256, 256)
		self.hidden4 = nn.Linear(256, 256)
		self.out = nn.Linear(256, 5)

	def forward(self, x):
		# baseline
		# x = F.sigmoid(self.hidden1(x))
		# x = self.out(x)
		# improved
		x = self.hidden1(x)
		x = F.relu(x)
		x = self.hidden2(x)
		x = F.relu(x)
		x = self.hidden3(x)
		x = F.relu(x)
		x = self.hidden4(x)
		x = F.relu(x)
		x = self.out(x)
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		# baseline
		# self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		# self.pool = nn.MaxPool1d(kernel_size=2)
		# self.conv2 = nn.Conv1d(6, 16, 5)
		# self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		# self.fc2 = nn.Linear(128, 5)
		# improved
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(6, 16, 5)
		self.drop = nn.Dropout(p=0.2)
		self.fc1 = nn.Linear(in_features=16 * 41, out_features=128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 5)

	def forward(self, x):
		# baseline
		# x = self.pool(F.relu(self.conv1(x)))
		# x = self.pool(F.relu(self.conv2(x)))
		# x = x.view(-1, 16 * 41)
		# x = F.relu(self.fc1(x))
		# x = self.fc2(x)
		#improved
		x = self.pool(F.relu(self.drop(self.conv1(x))))
		x = self.pool(F.relu(self.drop(self.conv2(x))))
		x = x.view(-1, 16 * 41)
		x = F.relu(self.drop(self.fc1(x)))
		x = F.relu(self.drop(self.fc2(x)))
		x = self.fc3(x)
		return x


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		# baseline
		# self.rnn = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True, dropout = 0.5)
		# self.fc = nn.Linear(in_features=16, out_features=5)
		# improved
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=2, batch_first=True, dropout = 0.5)
		self.fc1 = nn.Linear(in_features=32, out_features=16)
		self.fc2 = nn.Linear(in_features=16, out_features=5)

	def forward(self, x):
		# baseline
		# x, _ = self.rnn(x)
		# x = self.fc(x[:, -1, :])
		# improved
		x, _ = self.rnn(x)
		x = self.fc1(x[:, -1, :])
		x = F.relu(x)
		x = self.fc2(x)
		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc1 = nn.Linear(in_features=dim_input, out_features=64)
		self.rnn = nn.LSTM(input_size=64, hidden_size=16, num_layers=2, batch_first=True, dropout= 0.5, bidirectional = False)
		# if bidirectional, next in_features * 2
		self.fc2 = nn.Linear(in_features=16, out_features=8)
		self.fc3 = nn.Linear(in_features=8, out_features=2)
		#self.fc4 = nn.Linear(in_features=8, out_features=2)
		#self.drop = nn.Dropout(p=0.2)

	def forward(self, input_tuple):
		# Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		seqs, lengths = input_tuple
		seqs = F.tanh(self.fc1(seqs))
		seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
		seqs, _ = self.rnn(seqs)
		seqs, _ = pad_packed_sequence(seqs, batch_first=True)
		seqs = seqs[np.arange(len(seqs)),lengths-1]
		seqs = F.relu(self.fc2(seqs))
		#seqs = F.relu(self.fc3(seqs))
		seqs = self.fc3(seqs)
		return seqs