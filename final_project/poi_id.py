#!/usr/bin/python2

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
items_to_drop = ['TOTAL','THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
for outlier in items_to_drop:
    data_dict.pop(outlier, 0)
    
### Task 3: Create new feature(s)
if_nan_make_zero = lambda x : 0 if x == 'NaN' else x 

for poi in data_dict:
    if data_dict[poi]['to_messages'] == 'NaN':
        data_dict[poi]['msg_to_poi_prec'] = 0
    else:
        data_dict[poi]['msg_to_poi_prec'] = if_nan_make_zero(data_dict[poi]['from_this_person_to_poi']) / \
                                if_nan_make_zero(data_dict[poi]['to_messages'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

clf = make_pipeline(StandardScaler(), GaussianNB())

dump_classifier_and_data(clf, my_dataset, features_list)