import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import math

from perceptron_regression import Perceptron
from data_utils_regression import load_csv, train_test_split

data = load_csv("data/dataset1_regression.csv", has_header=False)
train_data, test_data = train_test_split(data, train_fraction=0.7, seed=42)

print("train size:", len(train_data))
print("test size:", len(test_data))

X_train = [[row[0]] for row in train_data]
y_train = [row[1] for row in train_data]
X_test = [[row[0]] for row in test_data]
y_test = [row[1] for row in test_data]

model = Perceptron(n_inputs=1, activation="linear", seed=1, update_mode="batch")
error_history = model.train(X_train, y_train, error_threshold=0.001, max_epochs=100000)

print("epochs run:", len(error_history))
print("final training error:", error_history[-1])
print("weights:", model.w, "bias:", model.b)

# --- 1. Error vs epoch plot ---
plt.figure(figsize=(8,5))
plt.plot(range(1, len(error_history)+1), error_history)
plt.xlabel("Epoch")
plt.ylabel("Average Error")
plt.title("Univariate Regression - Error vs Epoch")
plt.savefig("results/error_vs_epoch.png", dpi=130, bbox_inches="tight")
plt.show()

# --- 2. RMSE and %RMSE ---
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

# --- 3. Model output vs target (overlaid), train and test ---
def plot_model_vs_target(X, y_true, y_pred, title, filename):
    sorted_data = sorted(zip(X, y_true, y_pred), key=lambda row: row[0][0])
    x_sorted = [row[0][0] for row in sorted_data]
    y_true_sorted = [row[1] for row in sorted_data]
    y_pred_sorted = [row[2] for row in sorted_data]

    plt.figure(figsize=(8,5))
    plt.scatter(x_sorted, y_true_sorted, s=10, label="Target (actual)", alpha=0.6)
    plt.plot(x_sorted, y_pred_sorted, color="red", linewidth=2, label="Model output (predicted)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.show()

plot_model_vs_target(X_train, y_train, train_predictions, "Model Output vs Target (Training Data)", "results/model_vs_target_train.png")
plot_model_vs_target(X_test, y_test, test_predictions, "Model Output vs Target (Test Data)", "results/model_vs_target_test.png")

# --- 4. Scatter plot: target vs predicted ---
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
