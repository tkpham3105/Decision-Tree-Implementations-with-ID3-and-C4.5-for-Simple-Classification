# -*- coding: utf-8 -*-
"""

DECISION_TREE_WITH_NAIVE_BAYES
@author: PHAM Trung Kien

"""
###############################################################################
# Import necessary libraries
import os
import pandas as pd
import time


###############################################################################
# Change the working directory to where the train and test data files are located
os.chdir("D:\\tkpham\\COMP4331\\Assignment_2")


###############################################################################
# Load the train data and test data files as dataframes using pandas 
df_train = pd.read_csv("train.txt", sep = ',') # train data
df_test  = pd.read_csv("test.txt", sep = ',')  # test data
df_test_normalized = df_test.copy() # make a copy of test file to store the prediction result 


###############################################################################
"""
Helper function to obtain all different values of a column of a dataframe
    
    Parameters: atr_or_class - targeted attribute or class column
                dataframe - targeted dataframe
    
    Return: different values
"""
def get_all_values(atr_or_class, dataframe):
    all_values = list(dataframe[atr_or_class])
    all_values = list(dict.fromkeys(all_values)) # remove all duplicates: [a,a,b,b,c] -> [a,b,c]
    return all_values

list_different_classes = get_all_values('NURSERY', df_train)


###############################################################################
"""
This part is to nomalize the dataset (both train and test dataframes):
    Index-based normalization by the lists of different values respectively to each column 
    obtained from the function get_all_values above
    Eg. Consider column A of dataframe D: get_all_values(A,D) = [a,b,c,d]
        And if the original column A is:    [a,a,a,b,b,d,d,c] 
        After normalized, column A becomes: [0,0,0,1,1,3,3,2]

Normalization will help simplify the datasets and make them easier to be manipulated later on
Notice: Normalization should always take the train dataset as reference. In other words, we 
        should normalize other datasets based on the attributes or class values of the train dataset         
"""
for atr_or_class in df_test_normalized.keys():
    df_test_normalized[atr_or_class] = df_test_normalized[atr_or_class].apply(lambda x: get_all_values(atr_or_class, df_train).index(x))

for atr_or_class in df_train.keys():
    df_train[atr_or_class] = df_train[atr_or_class].apply(lambda x: get_all_values(atr_or_class, df_train).index(x))

###############################################################################
# Group the train dataframe by values of class column to calculate prior probabilities
df_train_groups = df_train.groupby('NURSERY')


###############################################################################
"""
Function to calculate P(atr = atr_value | class_value)

	Parameters: atr - targeted attribute
				atr_value - value of attribute atr to consider 
				class_value - value of class column to justify

	Return: calculation result of P(atr = atr_value | class_value) 
"""
def cal_prob_atr(atr, atr_value, class_value):

	# since the train dataframe has been normalized, we need to find index of class_value in list_different_classes
	# to count frequency of atr_value according to class_value subsequently 
	index_class_value = list_different_classes.index(class_value)

	# consider only small dataframe of corresponding class_value
	df_class_value = [df for v, df in df_train_groups if v == index_class_value][0]

	# count frequency of atr_value in atr column of df_class_value 
	freq_count = list(df_class_value[atr]).count(atr_value)

	if freq_count > 0:
		return freq_count/len(df_class_value) # needed probability

	# need to avoid the case when the probability = 0
	else:
		return 0.001/(len(df_class_value)+0.001) # perform Laplacian correction (we can add 1 but for use 0.001 instead for closer estimation)


###############################################################################
"""
Function to calculate P(atr_combination | class_value):

	Parameters: atr_combination - targeted combination of attributes' values: Eg. (atr_A = a1, atr_B = b1, atr_C = c1)
				class_value - value of class column to justify

	Return: calculation result of P(atr_combination | class_value)
			Eg. P(atr_combination | class_value) = P(atr_A = a1 | class_value)*P(atr_B = b1 | class_value)*P(atr_C = c1 | class_value) 
"""
def cal_prob_combi(atr_combination, class_value):
	probability = 1
	for i, atr in enumerate(df_test_normalized.keys()):
		probability *= cal_prob_atr(atr, atr_combination[i], class_value)
	return probability


###############################################################################
"""
Function to calculate P(atr_combination | class_value)*P(class_value) for all possible different class_value
	
	Parameters: atr_combination - targeted combination of attributes' values

	Return: calculation result of P(atr_combination | class_value)*P(class_value)
"""
def cal_final_probs_for_compare(atr_combination):
	dict_probability = {} # dictionary to store probabilities 
	len_train_df = len(df_train)
	for class_value in list_different_classes:

		# again, we need to find index of each class_value in list_different_classes
		class_value_index = list_different_classes.index(class_value)

		# calculate P(class_value)
		prob_class_val = list(df_train['NURSERY']).count(class_value_index)/len_train_df

		# calculate P(atr_combination | class_value)
		prob_combi = cal_prob_combi(atr_combination, class_value)

		# correspondingly store the result into dict_probability  
		dict_probability[class_value] = prob_combi*prob_class_val
		
	return dict_probability


###############################################################################
"""
Function to predict the class for target atr_combination

	Parameters: atr_combination - targeted combination of attributes' values

	Return: desired prediction
"""
def predict(atr_combination):
	dict_probability = cal_final_probs_for_compare(atr_combination)
	list_probs = [prob for _,prob in dict_probability.items()]

	# get the highest probability and return the corresponding class value
	# we only need to compare P(atr_combination | class_value)*P(class_value) between different class_value 
	prediction = [class_value for class_value,prob in dict_probability.items() if prob == max(list_probs)]
	return prediction	


###############################################################################
"""
Calculating prior probabilities and predicting parts:
	- Calculate needed probabilities
	- Predict class value
	- Calculate the runtime and export the results to excel file
"""
start = time.time()
df_test['Prediction'] = df_test_normalized.apply(lambda x: predict([x[atr] for atr in list(x.keys())]), axis = 1)
print(time.time()-start)
df_test.to_excel('Prediction_naive_bayes.xlsx')
print(df_test)


###############################################################################


