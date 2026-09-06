import math

def compute_rmse(y_true,y_pred):
  n=len(y_true)
  mse=sum((y_true[i]-y_pred[i])**2 for i in range(n))
  return math.sqrt(mse)

def compute_percent_rmse(y_true,y_pred):
  rmse = compute_rmse(y_true,y_pred)
  y_range=max(y_true)-min(y_true)
  return (rmse/y_range)*100
