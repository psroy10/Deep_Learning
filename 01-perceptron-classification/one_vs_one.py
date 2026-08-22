from perceptron import Perceptron

def prepare_pair_data(all_data,class_a,class_b,activation):
  X=[]
  Y=[]
  for row in all_data:
    features=list(row[:-1])
    label= row[-1]

    if label == class_a:
      X.append(features)
      Y.append(1)
    elif label == class_b:
      X.append(features)
      Y.append(0 if activation == "sigmoid" else -1)
      
  return X,Y

def one_vs_one (all_data,class_labels,activation="sigmoid",update_mode ="online",error_threshold=0.001,max_epochs=100000):

  trained_models={}
  error_histories={}

  n_inputs = len(all_data[0])-1

  for i in range(len(class_labels)):
    for j in range(i+1,len(class_labels)):
      class_a=class_labels[i]
      class_b=class_labels[j]

      X_train,y_train=prepare_pair_data(all_data,class_a,class_b,activation)
      

      model=Perceptron(n_inputs=n_inputs,activation=activation,update_mode=update_mode)
      errors=model.train(X_train,y_train,error_threshold,max_epochs)

      trained_models[(class_a,class_b)]=model
      error_histories[(class_a,class_b)]=errors

      print(f"Trained pair {class_a} vs {class_b} -> epochs: {len(errors)}, final error: {errors[-1]:.5f}")
  return trained_models,error_histories
def decode_vote(raw_prediction,class_a,class_b,activation):
  return class_a if raw_prediction ==1 else class_b


def predict_one_vs_one(x,trained_models,activation):
  votes={}

  for(class_a,class_b),model in trained_models.items():
    raw_prediction = model.predict_label(x)
    predicted_class=decode_vote(raw_prediction,class_a,class_b,activation)

    if predicted_class in votes:
      votes[predicted_class]+=1
    else :
      votes[predicted_class]=1
    

  best_class=None
  best_count=-1

  for class_label, count in votes.items():
    if count>best_count:
      best_count=count
      best_class=class_label
  return best_class

def get_predictions(data,trained_models,activation):
  true_labels=[]
  predicted_labels=[]
  for row in data:
    x=list(row[:-1])
    true_label=row[-1]

    predicted=predict_one_vs_one(x,trained_models,activation)

    true_labels.append(true_label)
    predicted_labels.append(predicted)

  return true_labels,predicted_labels
