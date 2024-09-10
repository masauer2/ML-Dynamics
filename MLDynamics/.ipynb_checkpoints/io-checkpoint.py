import pandas as pd
import numpy as np

def select_atom(df, col):
    return df[col]

def select_mode(df, modeNum = 0):
    return_df = df.copy()
    nRows = np.shape(df)[0]
    
    if len(np.shape(df)) != 1:
        nColumns = np.shape(df)[1]
        for row in range(nRows):
            for col in range(nColumns):
                return_df.iloc[row,col] = df.iloc[row,col][modeNum]
    else:
        nColumns = 1
        for row in range(nRows):
            return_df.iloc[row] = df.iloc[row][modeNum]

    return return_df

def add_labels(df, label, label_arr):
    
    # Shallow copy
    return_df = df.copy()

    # Number of rows will determine how many labels to add
    nRows = np.shape(df)[0]

    # We need at least one label to add
    nLabelsToAdd = 1

    # Check to see if we have multiple labels 
    # TO FIX: Throw error if number of labels does not equal number of label_arr provided
    if len(label) > 1 or len(label_arr) > 1:
        nLabelsToAdd = np.min([len(label), len(label_arr)])

    # Add each label - assume labels are in same order as dataframe rows
    for i in range(nLabelsToAdd):
        if nRows != len(label_arr[i]):
            label_arr[i] = label_arr[i][:nRows]
        return_df[label[i]] = label_arr[i]
        
    return return_df