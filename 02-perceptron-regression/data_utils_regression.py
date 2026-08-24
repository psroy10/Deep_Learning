import csv
import random

def load_csv(filepath, has_header=False):
  data=[]
  with open(filepath, "r") as f:
    reader = csv.reader(f)
    if has_header:
      next(reader)
    for row in reader:
      features = [float (v) for v in row[:-1]]
      target = float (row[-1])
      data.append(tuple(features) +(target,))
  return data

def train_test_split(data,train_fraction=0.7,seed=42):
  random.seed(seed)
  data = data[:]
  random.shuffle(data)
  n_train=int(train_fraction*len(data));
  return data[:n_train],data[n_train:]
