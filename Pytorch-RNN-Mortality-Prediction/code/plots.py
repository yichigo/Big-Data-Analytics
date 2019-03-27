import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.title('Loss Curve')
	plt.legend(loc="upper right")
	plt.savefig('MyVariableRNN_Loss.png')

	plt.figure()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
	plt.ylabel('Accuracy')
	plt.xlabel('epoch')
	plt.title('Accuracy Curve')
	plt.legend(loc="upper left")
	plt.savefig('MyVariableRNN_Accuracy.png')


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	
	y_true, y_pred = zip(*results)
	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	# Only use the labels that appear in the data
	#class_names = class_names[unique_labels(y_true, y_pred)]
	
	np.set_printoptions(precision=2)
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]), 
		yticks=np.arange(cm.shape[0]),
		# ... and label them with the respective list entries
		xticklabels=class_names, yticklabels=class_names,
		title='Normalized Confusion Matrix',
		ylabel='True',
		xlabel='Predicted')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt), 
				ha="center", va="center", 
				color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()

	plt.savefig("MyVariableRNN_Confusion_Matrix.png")









