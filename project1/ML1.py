from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

dia = datasets.fetch_openml(data_id=4134)
do = datasets.fetch_openml(data_id=41964)


def main():
  
  mytree2 = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=200)
  scores = model_selection.cross_val_predict(mytree2, dia.data, dia.target, method="predict_proba", cv=10)
  fpr, tpr, th = roc_curve(dia.target, scores[:,1], pos_label="1")
  fig, ax = plt.subplots() 
  ax.plot(fpr, tpr, label='min_sample_leaf = 110')
  ax.set_xlabel('1 - Specificity') 
  ax.set_ylabel('Sensitivity')
  ax.set_title('ROC curve for Bioresponse') 
  ax.legend() 
  plt.show()
  

def main2():
  
  x = [1,5,10,50,110,200,1000,2000]
  roc = []
  for i in x:
    mytree2 = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=i)
    y_scores2 = model_selection.cross_val_predict(mytree2, dia.data, dia.target, method="predict_proba", cv=10)
    roc.append(roc_auc_score(dia.target, y_scores2[:,1]))
    print(roc_auc_score(dia.target, y_scores2[:,1]))
  
  plt.plot(x,roc,'s-',color = 'r')
  plt.xlabel("min_sample_leaf")
  plt.ylabel("AUC score")
  
  plt.show()
  

def main3():

  tree = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=40)
  scores2 = model_selection.cross_val_predict(tree, do.data, do.target, method="predict_proba", cv=10)
  fpr, tpr, th = roc_curve(do.target, scores2[:,1], pos_label="1")
  fig, ax = plt.subplots() 
  ax.plot(fpr, tpr, label='min_sample_leaf = 40')
  ax.set_xlabel('1 - Specificity') 
  ax.set_ylabel('Sensitivity')
  ax.set_title('ROC curve for USPS') 
  ax.legend() 
  plt.show()

def main4():
  
  x = [1,10,40,110,200,400,600,1000]
  roc2 = []
  for i in x:
    mytree = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=i)
    y_scores = model_selection.cross_val_predict(mytree, do.data, do.target, method="predict_proba", cv=10)
    roc2.append(roc_auc_score(do.target, y_scores[:,1]))
    print(roc_auc_score(do.target, y_scores[:,1]))
  plt.plot(x,roc2,'s-',color = 'r')
  plt.xlabel("min_sample_leaf")
  plt.ylabel("AUC score")
  
  plt.show()

print('this is data 1') 
main()
main2()
print('this is data 2') 
main3()
main4()

