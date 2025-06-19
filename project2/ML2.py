from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def category_feature(dia):
  category_features = []
  for column in dia.data.columns:
    if dia.data[column].dtype == 'category':
        category_features.append(column)
  
  #print("Category features:", category_features)
  return category_features

def MinMax(data):
    scaler = MinMaxScaler() #实例化
    data[data.columns] = scaler.fit_transform(data[data.columns]) #fit_transform是trainData
    

def onehotencoding(dia,category_features):
    ohe = OneHotEncoder(sparse_output=False)
    data = dia.data.copy()
    
    for feature in category_features:
      encode = dia.data[[feature]]
      encoded = ohe.fit_transform(encode)
      encoded_feature_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([feature]))
      data = pd.concat([data, encoded_feature_df], axis=1)
      data = data.drop(columns=[feature])
    #print(data)
    MinMax(data)
    return data
    
def Decisiontree(data,dia):
    dtc = DecisionTreeClassifier()
    parameters = [{'min_samples_leaf':[10,30,50,100,200,150,250]}]
    tuned_dtc = GridSearchCV(dtc,parameters,scoring='roc_auc',cv=5)
    tuned_dtc.fit(data,dia.target)
    print("Decisiontree Best parameters found: ", tuned_dtc.best_params_)
    cv_result = cross_val_score(tuned_dtc, data, dia.target, cv=10, scoring='roc_auc')
    #print("Decisiontree means : ",cv_result.mean())
    #print("Decisiontree standard deviations : ",cv_result.std())
    return cv_result.mean(), cv_result.std()

def KNN(data,dia):
    knn = KNeighborsClassifier()
    param_grid  =  dict ( n_neighbors = (10,30,50,100,200)) 
    tuned_knn = GridSearchCV(knn,param_grid,scoring='roc_auc',cv=5)
    tuned_knn.fit(data,dia.target)
    print("KNN Best parameters found: ", tuned_knn.best_params_)
    knn_cv_result = cross_val_score(tuned_knn, data, dia.target, cv=10, scoring='roc_auc')
    #print("KNN means : ",knn_cv_result.mean())
    #print("KNN standard deviations : ",knn_cv_result.std())
    return knn_cv_result.mean(), knn_cv_result.std()

def naive_bayes(data,dia):
    naive = MultinomialNB()
    naive.fit(data,dia.target)
    naive_cv_result = cross_val_score(naive, data, dia.target, cv=10, scoring='roc_auc')
    #print("naive_bayes mean: ",naive_cv_result.mean())
    #print("naive_bayes standard deviations : ",naive_cv_result.std())
    return naive_cv_result.mean(), naive_cv_result.std()

def logistic(data,dia):
    logistic = LogisticRegression()
    logistic.fit(data,dia.target)
    logistic_cv_result = cross_val_score(logistic, data, dia.target, cv=10, scoring='roc_auc')
    #print("logistic mean: ",logistic_cv_result.mean())
    #print("logistic standard deviations : ",logistic_cv_result.std())
    return logistic_cv_result.mean(), logistic_cv_result.std()

def dummy(data,dia):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(data,dia.target)
    dummy_cv_result = cross_val_score(dummy, data, dia.target, cv=10, scoring='roc_auc')
    #print("dummy mean: ",dummy_cv_result.mean())
    #print("dummy standard deviations : ",dummy_cv_result.std())
    return dummy_cv_result.mean(), dummy_cv_result.std()

def evaluate_models(data, dia):
    results = {
        "Model": ["Decision Tree", "KNN", "Naive Bayes", "Logistic Regression", "Dummy"],
        "Mean AUC": [],
        "Standard Deviation AUC": []
    }
    # Collect results from each model
    dt_mean, dt_std = Decisiontree(data, dia)
    knn_mean, knn_std = KNN(data, dia)
    nb_mean, nb_std = naive_bayes(data, dia)
    lr_mean, lr_std = logistic(data, dia)
    dummy_mean, dummy_std = dummy(data, dia)
    
    # Organize results into a numpy array
    results["Mean AUC"].extend([dt_mean, knn_mean, nb_mean, lr_mean, dummy_mean])
    results["Standard Deviation AUC"].extend([dt_std, knn_std, nb_std, lr_std, dummy_std])
    
    # Convert to DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    return results_df

def main():
    
    print("Results for Dataset 1:")
    dia = datasets.fetch_openml(data_id=41283)
    category_features = category_feature(dia)
    data = onehotencoding(dia,category_features)
    results_array = evaluate_models(data, dia)
    print(results_array)

def main2():
    
    print("Results for Dataset 2:")
    dia2 = datasets.fetch_openml(data_id=41335)
    category_features2 = category_feature(dia2)
    onehotencoding(dia2,category_features2)
    data = onehotencoding(dia2,category_features2)
    results_array = evaluate_models(data, dia2)
    print(results_array)


main()
main2()
