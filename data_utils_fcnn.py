import csv
import random

def load_csv(filepath,has_header=True):
  data=[]
  with open(filepath, "r") as f:
    reader = csv.reader(f)
    if has_header:
      next(reader)
    for row in reader:
      features=[float(v) for v in row[:-1]]
      label=row[-1]
      data.append(tuple(features)+(label,))
  return data


def train_val_test_split(data,train_fraction=0.6,val_fraction=0.2,seed=42):
  random.seed(seed)
  data=data[:]
  random.shuffle(data)

  n=len(data)
  n_train=int(train_fraction*n)
  n_val=int(val_fraction*n)

  train_data=data[:n_train]
  val_data=data[n_train:n_train+n_val]
  test_data=data[n_train+n_val:]


  return train_data,val_data,test_data


def stratified_train_val_test_split(data,train_fraction=0.6,val_fraction=0.2,seed=42):
  random.seed(seed)

  grouped={}
  for row in data:
    label=row[-1]
    grouped.setdefault(label,[]).append(row)

  train_data,val_data,test_data=[],[],[]


  for label,rows in grouped.items():
    rows= rows[:]
    random.shuffle(rows)
    n=len(rows)
    n_train=int(train_fraction*n)
    n_val=int(val_fraction*n)

    train_data.extend(rows[:n_train])
    val_data.extend(rows[n_train:n_train+n_val])
    test_data.extend(rows[n_train+n_val:])


  random.shuffle(train_data)
  random.shuffle(val_data)
  random.shuffle(test_data)

  return train_data,val_data,test_data
