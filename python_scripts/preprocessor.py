import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
import pydot
from sklearn.model_selection import train_test_split
from sklearn import metrics
from statistics import mean

# Data Cleaning
leResult = preprocessing.LabelEncoder()
leTeamName = preprocessing.LabelEncoder()

def merge_get_data():
    # Merge 3 dataset into one dataframe and reset Index
    laliga2019 = pd.read_csv('Data/LaLigaP-20192020.csv')
    laliga2020 = pd.read_csv('Data/LaLigaP-20202021.csv')
    laliga2021 = pd.read_csv('Data/LaLigaP-20212022.csv')
    print(laliga2019.shape)
    print(laliga2020.shape)
    print(laliga2021.shape)
    frames = [laliga2019, laliga2020, laliga2021]
    laliga = pd.concat(frames)
    return laliga

def displayNullAndDrop(data):
    # Print all the null value inside the dataframe
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data.isnull().sum()[data.isnull().sum() > 0])
    # Clean the row which contain missing value
    data = data.dropna()
    print("Null value contained row deleted.")
    return data

# Split the dataset which only contain key result and match statistics data (Without betting odds data)
def SplitResultMatchStat(data):
    columns_req = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                   'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    laliga_Result_Match_Statistics = data[columns_req]
    return laliga_Result_Match_Statistics

def meanOfOdds(laliga):
    columnH = ['B365H', 'BWH', 'IWH', 'PSH', 'WHH', 'VCH']
    laliga['MOH'] = laliga[columnH].mean(axis=1)
    columnA = ['B365D', 'BWD', 'IWD', 'PSD', 'WHD', 'VCD']
    laliga['MOD'] = laliga[columnA].mean(axis=1)
    columnA = ['B365A', 'BWA', 'IWA', 'PSA', 'WHA', 'VCA']
    laliga['MOA'] = laliga[columnA].mean(axis=1)
    return laliga

def encode_string(laliga):
    # Encode string data to numerical data by LabelEncoder
    laliga.loc[:, ['HomeTeam', 'AwayTeam']] = laliga.loc[:, ['HomeTeam', 'AwayTeam']].apply(
        leTeamName.fit_transform)
    laliga.loc[:, ['HTR', 'FTR']] = laliga.loc[:, ['HTR', 'FTR']].apply(leResult.fit_transform)
    return laliga

def decode_ftr(finalDataset):
    finalDataset['FTR'] = leResult.inverse_transform(finalDataset['FTR'])
    return finalDataset