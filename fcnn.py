import math
import random

class FCNN:
  def __init__(self,n_inputs,hidden_layers,n_outputs,min_input,max_input,hidden_activation="sigmoid",output_activation="sigmoid",seed=1):
    random.seed(seed)
    self.n_inputs=n_inputs
    self.hidden_layers=hidden_layers
    self.n_outputs=n_outputs
    self.hidden_activation=hidden_activation
    self.output_activation=output_activation
    self.min_input=min_input
    self.max_input=max_input
    self.weights=[]
    self.initialize_weights()

  def initialize_weights(self):
    layers_sizes=[self.n_inputs]+self.hidden_layers+[self.n_outputs]
    for i in range(len(layers_sizes)-1):
      n_inputs=layers_sizes[i]
      n_outputs=layers_sizes[i+1]
      layer_weights=[]
      for j in range(n_outputs):
        neuron_weights=[]
        neuron_weights.append(random.uniform(self.min_input,self.max_input))
        for k in range (n_inputs):
          neuron_weights.append(random.uniform(self.min_input,self.max_input))
        layer_weights.append(neuron_weights)
      self.weights.append(layer_weights)

  def apply_activation(self,net,activation_type):
    if activation_type == "sigmoid":
      return 1/(1+math.exp(-net))
    elif activation_type == "tanh":
      return math.tanh(net)
    elif activation_type == "linear":
      return net

  def compute_net_input(self,x,neuron_weights):
    total=neuron_weights[0]
    for i in range(len(x)):
      total+=neuron_weights[i+1]*x[i]
    return total

  def forward_layer(self,x,layer_weights,activation_type):
    outputs = []
    for neuron_weights in layer_weights:
      net=self.compute_net_input(x,neuron_weights)
      out=self.apply_activation(net,activation_type)
      outputs.append(out)
    return outputs

  def forward(self,x):
    layer_outputs=[]
    current_output=x
    num_layers=len(self.weights)
    for i,layer_weights in enumerate(self.weights):
      is_output_layer=(i==num_layers-1)
      activation_type= self.output_activation if is_output_layer else self.hidden_activation
      current_output = self.forward_layer(current_output,layer_weights,activation_type)
      layer_outputs.append(current_output)
    return current_output,layer_outputs

  def activation_derivative(self, out,activation_type):
    if activation_type == "sigmoid":
      return out*(1-out)
    elif activation_type == "tanh":
      return 1-out**2
    elif activation_type == "linear":
      return 1

  def compute_error(self,target,output):
    error=0.0
    for i in range(len(target)):
      error+=0.5*(target[i]-output[i])**2
    return error

  def compute_output_deltas(self,target,output):
    deltas=[]
    for k in range(len(output)):
      error=target[k]-output[k]
      deriv=self.activation_derivative(output[k],self.output_activation)
      delta=error*deriv
      deltas.append(delta)
    return deltas

  def compute_hidden_deltas(self,next_layer_deltas,next_layer_weights,current_layer_outputs):
    deltas=[]
    n_current_neurons=len(current_layer_outputs)
    for j in range(n_current_neurons):
      weighted_sum=0.0
      for k in range(len(next_layer_deltas)):
        weighted_sum+=next_layer_deltas[k]*next_layer_weights[k][j+1]
      deriv=self.activation_derivative(current_layer_outputs[j],self.hidden_activation)
      delta=weighted_sum*deriv
      deltas.append(delta)
    return deltas

  def train_one_example(self,x,target,learning_rate):
    output,layer_outputs=self.forward(x)

    all_deltas=[None]*len(self.weights)
    all_deltas[-1]=self.compute_output_deltas(target,output)

    for layer_index in range(len(self.weights)-2,-1,-1):
      next_layer_deltas=all_deltas[layer_index+1]
      next_layer_weights=self.weights[layer_index+1]
      current_layer_outputs=layer_outputs[layer_index]
      all_deltas[layer_index]=self.compute_hidden_deltas(next_layer_deltas,next_layer_weights,current_layer_outputs)

    for layer_index in range(len(self.weights)):
      if layer_index == 0:
        layer_input=x
      else:
        layer_input=layer_outputs[layer_index-1]

      deltas=all_deltas[layer_index]
      layer_weights=self.weights[layer_index]

      for neuron_index in range(len(layer_weights)):
        neuron_weights = layer_weights[neuron_index]
        delta=deltas[neuron_index]
        neuron_weights[0]+=learning_rate*delta
        for i in range(len(layer_input)):
          neuron_weights[i+1]+=learning_rate*delta*layer_input[i]

    error=self.compute_error(target,output)
    return error

  def train(self,X,y,learning_rate=0.1,max_epoches=1000,error_threshold=0.001):
    error_history=[]
    n=len(X)

    for epoch in range(max_epoches):
      indices=list(range(n))
      random.shuffle(indices)

      total_error=0.0
      for idx in indices:
        x=X[idx]
        target=y[idx]
        error=self.train_one_example(x,target,learning_rate)
        total_error+=error

      avg_error=total_error/n
      error_history.append(avg_error)

      if avg_error < error_threshold:
        break

    return error_history
  def predict(self, x):
    output, _ = self.forward(x)
    return output

