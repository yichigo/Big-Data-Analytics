import utils
import etl
import models_partc

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

RANDOM_STATE = 545510477

#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
	X_train, Y_train = utils.get_data_from_svmlight('../deliverables/features_svmlight.train')

	events_test = pd.read_csv('../data/test/events.csv')
	feature_map_test = pd.read_csv('../data/test/event_feature_map.csv')
	
	deliverables_path = '../deliverables/'
	aggregated_events_test = etl.aggregate_events(events_test, None, feature_map_test, deliverables_path)
	
	patient_features_test = aggregated_events_test.groupby('patient_id')[['feature_id','feature_value']]
	patient_features_test = patient_features_test.apply(lambda g: list(map(tuple, g.values.tolist()))).to_dict()
	
	op_file = deliverables_path + 'features_svmlight.test'
	op_deliverable = deliverables_path + 'test_features.txt'
	deliverable1 = open(op_file,'wb')
	deliverable2 = open(op_deliverable, 'wb')

	line1 = line2 = ''
	for key in sorted(patient_features_test.keys()):
		line1 +='1 '
		line2 += str(int(key)) +' '

		for value in sorted(patient_features_test[key]):
			line1 += str(int(value[0])) + ':' + str("{:.6f}".format(value[1])) + ' '
			line2 += str(int(value[0])) + ':' + str("{:.6f}".format(value[1])) + ' '

		line1 += '\n' 
		line2 += '\n'

	deliverable1.write(bytes(line1,'UTF-8')) #Use 'UTF-8'
	deliverable2.write(bytes(line2,'UTF-8'))  
	
	X_test = load_svmlight_file(deliverables_path + 'features_svmlight.test', n_features = 3190)[0]
	return X_train, Y_train, X_test

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this

	# Random Forest Model, grid search to find best parameter
	#rfc = RandomForestClassifier(random_state = RANDOM_STATE, oob_score=True, n_jobs = -1,
	#							 max_features='log2')
	#param_grid = {'n_estimators': [300, 400, 500, 600, 700], 'max_depth': [50,60,70,80,90,100]}
	#CV_rfc = GridSearchCV(estimator = rfc, param_grid = param_grid, n_jobs = -1,
	#					   cv = 10, verbose = 20, scoring = 'roc_auc')
	#CV_rfc.fit(X_train, Y_train)
	#print(CV_rfc.best_params_)
	#print(CV_rfc.best_score_)

	model = RandomForestClassifier(random_state = RANDOM_STATE, oob_score = True, n_jobs = -1,
								max_features='log2', n_estimators = 500, max_depth = 80)
	model.fit(X_train,Y_train)

	Y_pred = model.predict(X_test).astype(int) # returns integer 0 or 1
	#Y_pred = model.predict_proba(X_test)[:, 1] # returns probability, float in [0,1]
	return Y_pred


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
	main()

	