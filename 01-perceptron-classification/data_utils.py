import csv
import random

def load_csv(filepath, has_header=True):
  data=[]
  with open(filepath, "r") as f:
    reader = csv.reader(f)
    if has_header:
      next(reader)
    for row in reader:
      features = [float (v) for v in row[:-1]]
      label = int (row[-1])
      data.append(tuple(features) +(label,))
  return data

def train_test_split(data,train_fraction=0.7,seed=42):
  random.seed(seed)

  grouped = {}
  for row in data:
    label=row[-1]
    grouped.setdefault(label,[]).append(row)
  
  train_data=[]
  test_data=[]

  for label, rows in grouped.items():
    rows=rows[:]
    random.shuffle(rows)
    n_train = int(train_fraction * len(rows))
    train_data.extend(rows[:n_train])
    test_data.extend(rows[n_train:])

  random.shuffle(train_data)
  random.shuffle(test_data)

  return train_data,test_data
