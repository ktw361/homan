import os
import pandas as pd
import numpy as np

def get_results(workdir):
    dfs = []
    for root, dirs, files in os.walk(workdir):
        for file in files:
            if file.endswith('csv'):
                # print(root, file)
                df = pd.read_csv(os.path.join(root, file))
                dfs.append(df)
    df = pd.concat(dfs)
    return df

df = get_results('/home/skynet/Zhifan/homan-master/results/arctic_stable/samples')

print(len(df))
print(df.mean())
# printdf
