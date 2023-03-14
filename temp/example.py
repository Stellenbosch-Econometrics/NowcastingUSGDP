
import pandas as pd
import matplotlib.pyplot as plt
import torch

df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE.csv')
df['ds'] = pd.to_datetime(df['ds'])
df.head()