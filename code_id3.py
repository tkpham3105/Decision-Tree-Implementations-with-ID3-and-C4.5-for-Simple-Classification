# -*- coding: utf-8 -*-
"""

DECISION_TREE_WITH_ID3
@author: PHAM Trung Kien

"""

###############################################################################
# Import necessary libraries
import os
import math
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
"""
Function to calculate Info(Attribute,T) of an attribute of specified dataframe
    
    Parameters: atr - targeted attribute
                dataframe - targeted dataframe

    Return: result of Info(Attribute,T)

Notice: Please read file SUPPORT for more details with visualization  
"""
def calculate_Info_A_T(atr, dataframe):
    
    # consider only two columns of given atr and class -> create new dataframe with that two columns' data
    df_atr = pd.DataFrame(zip(dataframe[atr], dataframe['NURSERY']), columns = [atr,'NURSERY'])
    n = len(df_atr) 

    # firstly, group the above dataframe by values of the atr column. Thanks to normalization step, we can 
    # accurately perform grouping much faster  
    group_by_atr = df_atr.groupby(atr)

    Info_atr = 0 # variable to store return value

    for _, df_atr in group_by_atr:
        entropy = 0
        n1 = len(df_atr)

        # secondly, group each group_atr by values of the class column.  
        group_by_class = df_atr.groupby('NURSERY')

        for _, df_class in group_by_class:
            n2 = len(df_class)

            # calculation steps
            p = n2/n1
            if p == 0: # if p = 0, continue since p*log(p,2)->0 as p->0 and to avoid error caused by math.log(0,2) 
                continue
            else:
                entropy += -p*math.log(p, 2)
        Info_atr += entropy*(n1/n)
    return Info_atr


###############################################################################
"""
Function to calculate Info(T) of given dataframe
    
    Parameter: dataframe - targeted dataframe

    Return: result of Info(T)
"""    
def calculate_Info_T(dataframe):
    Info_T = 0
    list_labels = get_all_values('NURSERY', df_train)
    list_all_classes = list(dataframe['NURSERY'])
    for value in list_labels:

        # simply count apprearance frequency of each value in class column in order to calculate p
        p = list_all_classes.count(value)/len(list_all_classes)
        if p == 0:
            continue
        else:
            Info_T += -p*math.log(p,2)
    return Info_T


###############################################################################
"""
Function to calculate Gain(Attribute,T) of given dataframe
    
    Parameters: atr - targeted attribute  
                dataframe - targeted dataframe

    Return: result of Gain(Attribute,T) = Info(T) - Info(Attribute,T)
"""  
def calculate_Gain(atr, dataframe):
    return calculate_Info_T(dataframe) - calculate_Info_A_T(atr, dataframe)


############################################################################### 
"""
Class of Decision Tree's Nodes
Each Node will contain:
    - atr: the name of attribute use for value-based splitting, assigned None for every leaf nodes
    - childer: the dictionary contains every children nodes with keys are atr's values
    - label: the class value it belongs to, assigned None when the node is not a leaf node
    - not_leaf: consider whether the node is a leaf node or not  
"""
class Node:    
    def __init__(self):
        self.atr = None
        self.children = {}
        self.label = None
        self.not_leaf = True


############################################################################### 
"""
Recursive training function to construct Decision Tree recursively:
    
    Parameters: dataframe - targeted dataframe
                node - track node (assigned root at first)  
"""
def train_id3(dataframe, node):

    # if there is no attribute remaining (only has class column), perform majority voting
    if len(list(dataframe.keys())) == 1:
        dict_count = {}

        # count frequency of all values in class column
        for value in list_different_classes:
            dict_count[value] = dataframe['NURSERY'].count(value)

        # majority voting to find the value that has the highest apprearance frequency
        max_count = max([count for _,count in dict_count.items()])
        max_class_value = [class_value for class_value,count in dict_count.items() if count == max_count][0]

        # update the label of the node
        node.label = max_class_value
        node.children[value].not_leaf = False

    # if the node is not leaf and there are samples left and there are some remaining attributes for further partitioning, keep training
    if node.not_leaf and len(dataframe) != 0 and len(list(dataframe.keys())) > 1:

        # calculate Gain(Attribute, T) values for all attributes 
        dict_Gain = {}
        list_atr = list(dataframe.keys())
        for atr in list_atr[:len(list_atr)-1]:
            dict_Gain[atr] = calculate_Gain(atr, dataframe)

        # get the attribute with highest Gain(Attribute, T) value
        max_Gain = max([v for _,v in dict_Gain.items()])
        max_Gain_atr = [k for k,v in dict_Gain.items() if v == max_Gain]
        node.atr = max_Gain_atr[0] # assign max attribute to atr of current node

        # group the given dataframe by values of the max attribute
        group_by_max_atr = dataframe.groupby(node.atr)

        for value, df in group_by_max_atr:
            node.children[value] = Node() # create a new node as children node of current node

            # if Info(T) = 0, which means there is only one value in class column or no sample left in df,
            # then this children node is a leaf node   
            if calculate_Info_T(df) == 0: 
                node.children[value].not_leaf = False # update not_leaf to stop recursion
                df = df.reset_index() # reset index of df since groupby function does not do it 
                node.children[value].label = df['NURSERY'][0] # update label for prediction 

            # if not, continue perform training for this children node 
            else:
                df = df.drop(node.atr, axis = 1) # note to drop the column of current node's atr
                train_id3(df, node.children[value])


############################################################################### 
"""
Recusive function to recursively traverse the Decision Tree to finally generate prediction:
    
    Parameters: track_node - track node to traverse the tree
                test_data - combination of values of attributes to predict

    Return: Prediction class 
"""
def predict(track_node, test_data):

    # if track_node is currently not a leaf node
    if track_node.not_leaf:

        # find the index of the track_node's atr in the list of test dataframe's attributes
        split_atr_ind = list(df_test_normalized.keys()).index(track_node.atr)

        # update the track_node to its children node based on value of the corresponding 
        # attribute of the test_data
        track_node = track_node.children[test_data[split_atr_ind]]

        # recursion step until reaching the a leaf node
        return predict(track_node, test_data)

    # if track_node now is a leaf node, simply return its label
    else:
        prediction = track_node.label
        return prediction


###############################################################################
"""
Training and Testing parts of the program:
    - Create a root
    - Train the root based on the train dataset to expand the root recursively -> construct Decision Tree
    - Predict the classes for the test dataset
    - Calculate the runtime and export the results to excel file  
"""
root = Node()
start = time.time()
train_id3(df_train, root)
df_test['Prediction'] = df_test_normalized.apply(lambda x: predict(root, [x[atr] for atr in list(x.keys())]), axis = 1)
df_test['Prediction'] = df_test['Prediction'].apply(lambda x: list_different_classes[x])
print(time.time()-start)
df_test.to_excel('Prediction_id3.xlsx')
print(df_test)


###############################################################################
