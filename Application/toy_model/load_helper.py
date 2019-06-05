import pandas as pd
import numpy as np


# load a csv file to numpy tensor
def load_csv(path):
    file = pd.read_csv(path)
    return file.values


# save a numpy tensor to csv file
def save_csv(data, path):
    print(data)
    pd.DataFrame(data).to_csv(path)
