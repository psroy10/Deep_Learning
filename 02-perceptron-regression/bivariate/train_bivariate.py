import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

from perceptron_regression import Perceptron
from data_utils_regression import load_csv, train_test_split

data = load_csv("data/dataset2_regression.csv", has_header=False)
train_data, test_data = train_test_split(data, train_fraction=0.7, seed=42)

print("train size:", len(train_data))
print("test size:", len(test_data))

X_train = [[row[0],row[1]] for row in train_data]
y_train = [row[2] for row in train_data]
X_test = [[row[0],row[1]] for row in test_data]
y_test = [row[2] for row in test_data]

model = Perceptron(n_inputs=2, activation="linear", seed=1, update_mode="batch")
error_history = model.train(X_train, y_train, error_threshold=0.001, max_epochs=100000)

print("epochs run:", len(error_history))
print("final training error:", error_history[-1])
print("weights:", model.w, "bias:", model.b)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(error_history)+1), error_history)
plt.xlabel("Epoch")
plt.ylabel("Average Error")
plt.title("Bivariate Regression - Error vs Epoch")
plt.savefig("results/error_vs_epoch.png", dpi=130, bbox_inches="tight")
plt.show()

def compute_rmse(y_true, y_pred):
    n = len(y_true)
    mse = sum((y_true[i]-y_pred[i])**2 for i in range(n)) / n
    return math.sqrt(mse)

def compute_percent_rmse(y_true, y_pred):
    rmse = compute_rmse(y_true, y_pred)
    y_range = max(y_true) - min(y_true)
    return (rmse / y_range) * 100

train_predictions = [model.predict_label(x) for x in X_train]
test_predictions = [model.predict_label(x) for x in X_test]

train_rmse = compute_rmse(y_train, train_predictions)
test_rmse = compute_rmse(y_test, test_predictions)
train_percent_rmse = compute_percent_rmse(y_train, train_predictions)
test_percent_rmse = compute_percent_rmse(y_test, test_predictions)

print(f"\nTrain RMSE: {train_rmse:.4f} | Train %RMSE: {train_percent_rmse:.2f}%")
print(f"Test RMSE: {test_rmse:.4f} | Test %RMSE: {test_percent_rmse:.2f}%")

def plot_model_vs_target(X, y_true, y_pred, title, filename):
    x1 = [x[0] for x in X]
    x2 = [x[1] for x in X]
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1,x2,y_true,s=3,alpha=0.4,label="Target (actual)",color="blue")
    ax.scatter(x1,x2,y_pred,s=3,alpha=0.4,label="Model output (predicted)",color="red")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    ax.legend()
    plt.savefig(filename,dpi=130,bbox_inches="tight")
    plt.show()

plot_model_vs_target(X_train, y_train, train_predictions, "Model Output vs Target (Training Data)", "results/model_vs_target_train.png")
plot_model_vs_target(X_test, y_test, test_predictions, "Model Output vs Target (Test Data)", "results/model_vs_target_test.png")

plt.figure(figsize=(6,6))
plt.scatter(y_train, train_predictions, s=10, alpha=0.5, label="Train")
plt.scatter(y_test, test_predictions, s=10, alpha=0.5, label="Test", color="orange")
min_val = min(min(y_train), min(y_test))
max_val = max(max(y_train), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", label="Perfect prediction")
plt.xlabel("Target output (actual y)")
plt.ylabel("Model output (predicted y)")
plt.title("Target vs Predicted")
plt.legend()
plt.savefig("results/target_vs_predicted_scatter.png", dpi=130, bbox_inches="tight")
plt.show()
