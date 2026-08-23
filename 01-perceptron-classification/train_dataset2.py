from data_utils import load_csv
from one_vs_one import one_vs_one,get_predictions
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay

train_data=load_csv("data/dataset2_train.csv",has_header=True)
test_data=load_csv("data/dataset2_test.csv",has_header=True)

print("train size :",len(train_data))
print("test size :",len(test_data))

CLASS_LABELS=[1,2,3]

print("\n----training sigmoid----")
models_sigmoid,errors_sigmoid=one_vs_one(train_data,CLASS_LABELS,activation="sigmoid",update_mode="online",error_threshold=0.001,max_epochs=10000)


print("\n----training tanh----")
models_tanh,errors_tanh=one_vs_one(train_data,CLASS_LABELS,activation="tanh",update_mode="online",error_threshold=0.001,max_epochs=10000)


def evaluate_and_plot(models,error_histories,activation,train_data,test_data):
  plt.figure(figsize=(8,5))
  for pair,errors in error_histories.items():
    plt.plot(range(1,len(errors)+1),errors,label =f"Class {pair[0]} vs {pair[1]}")
  plt.xlabel("Epoch")
  plt.ylabel("Average Error")
  plt.title(f"Dataset2 - Error vs Epoch({activation}")
  plt.legend()
  plt.savefig(f"dataset2_{activation}_error_vs_epoch.png",dpi=130, bbox_inches="tight")
  plt.show()

  true_labels,predicted_labels=get_predictions(test_data,models,activation)
  count=sum(1 for t,p in zip(true_labels,predicted_labels) if t==p)
  accuracy=count/len(true_labels)
  print(f"\n{activation} test accuracy: {accuracy:.4f}")

  cm=confusion_matrix(true_labels,predicted_labels,labels=CLASS_LABELS)
  print("Confusion Matrix:")
  print(cm)
  print(classification_report(true_labels,predicted_labels,labels=CLASS_LABELS,digits=4))


  disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=CLASS_LABELS)
  disp.plot(cmap="Blues")
  plt.title(f"Confusion Matrix -Dataset2({activation})")
  plt.savefig(f"dataset2_{activation}_confusion_matrix.png", dpi=130, bbox_inches="tight")
  plt.show()



evaluate_and_plot(models_sigmoid,errors_sigmoid,"sigmoid",train_data,test_data)
evaluate_and_plot(models_tanh,errors_tanh,"tanh",train_data,test_data)

