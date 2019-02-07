import pandas as pd

data = pd.read_csv("./data.csv")

training=data.sample(frac=0.5) # Training set

data_rest = data.loc[~data.index.isin(training.index)]

testing = data_rest.sample(frac=0.5)
validation = data_rest.sample(frac=0.5)