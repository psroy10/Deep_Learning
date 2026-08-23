from data_utils import load_csv
from one_vs_one import one_vs_one, get_predictions

train_data=load_csv("data/dataset2_train.csv",has_header=True)
test_data=load_csv("data/dataset2_test.csv",has_header=True)

print("train size:",len(train_data))
print("test size:",len(test_data))
