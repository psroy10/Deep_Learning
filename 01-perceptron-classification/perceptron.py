

import math
import random

class Perceptron:
  def __init__(self,n_inputs,activation="sigmoid",seed=1,update_mode="online"):
    random.seed(seed)
    self.w=[random.uniform(-0.5,0.5) for _ in range(n_inputs)]
    self.b=random.uniform(-0.5,0.5)
    self.activation=activation
    self.update_mode=update_mode
    self.t=0

  
  def compute_net_input(self,x):
    total=self.b
    for i in range(len(self.w)):
      total+=self.w[i]*x[i]
    return total

  
  def apply_activation(self,net):
    if self.activation == "sigmoid":
      return 1/(1+math.exp(-net))
    elif self.activation == "tanh":
      return math.tanh(net)


  def activation_derivative(self,out):
    if self.activation == "sigmoid":
      return out*(1-out)
    elif self.activation == "tanh":
      return 1-out**2

  def predict_label(self,x):
    net = self.compute_net_input(x)
    out = self.apply_activation(net)
    if self.activation == "sigmoid":
      return 1 if out>=0.5 else 0
    elif self.activation == "tanh":
      return 1 if out>=0 else -1

  def batch_epoch(self,X,y,current_lr):
    n=len(X)
    dw=[0.0]*len(self.w)
    db=0.0
    total_squared_error=0.0

    for i in range(n):
      x=X[i]
      target =y[i]

      net = self.compute_net_input(x)
      out = self.apply_activation(net)

      error= target-out
      deriv=self.activation_derivative(out)
      delta = error*deriv

      for j in range(len(self.w)):
        dw[j]+=delta*x[j]
      db+=delta

      total_squared_error+= error**2

    for j in range(len(self.w)):
      self.w[j]+=current_lr*(dw[j]/n)
    self.b+=current_lr*(db/n)

    return total_squared_error/n


  def online_epoch(self,X,y,current_lr):
    n=len(X)
    total_squared_error=0.0

    for i in range(n):
      x=X[i]
      target=y[i]

      net=self.compute_net_input(x)
      out=self.apply_activation(net)

      error=target-out
      deriv=self.activation_derivative(out)
      delta = error * deriv

      for j in range(len(self.w)):
        self.w[j] +=current_lr*delta*x[j]
      self.b +=current_lr* delta

      total_squared_error+=error**2

    return total_squared_error/n



  def train(self,X,y,error_threshold=0.01,max_epochs=10000):
    error_history = []
    avg_error = float('inf')

    while avg_error>error_threshold and self.t<max_epochs:
      self.t+=1
      current_lr = 1.0/(1.0 + 0.001*self.t)

      if self.update_mode == "batch":
        avg_error= self.batch_epoch(X,y,current_lr)
      else:
        avg_error = self.online_epoch(X,y,current_lr)
      
      error_history.append(avg_error)

    return error_history
