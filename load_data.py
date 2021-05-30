import os
import numpy as np
import pandas as pd 


def load_json(json_file):
    with open(json_file, 'rb') as f:
        df = pd.read_json(f)
    return df


def Preprocessing(df):
    pass
    return df



if __name__ == '__main__':
    path = 'data/result2.json'
    df = load_json(path).drop(columns=['url', 'links'])
    df.to_csv('data2.csv', index=False)