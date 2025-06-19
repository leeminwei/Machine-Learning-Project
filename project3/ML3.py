from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

def preprocess_data(cpu):
    cpu.target = cpu.target.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(cpu.data, cpu.target, test_size=0.2, random_state=42)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return x_tr, x_val, x_test, y_tr, y_val, y_test

def model1(a,x_tr,x_val,y_tr,y_val,x_test,y_test):
    model1 = Sequential()
    model1.add(Input((a,)))
    model1.add(Dense(5,activation='sigmoid'))
    model1.add(Dense(1))
    model1.compile(loss='mse',optimizer='adam',metrics=['mse'])
    history = model1.fit(x_tr,y_tr,epochs=200,validation_data=(x_val,y_val),verbose=1)
    model1.evaluate(x_test,y_test,verbose=1)
    y_pred = model1.predict(x_test)
    return history,y_pred

def model2(b,x_tr,x_val,y_tr,y_val,x_test,y_test):

    model2 = Sequential()
    model2.add(Input((b,)))
    model2.add(Dense(30,activation='sigmoid'))
    model2.add(Dense(1))
    model2.compile(loss='mse',optimizer='adam',metrics=['mse'])
    history = model2.fit(x_tr,y_tr,epochs=200,validation_data=(x_val,y_val),verbose=1)
    model2.evaluate(x_test,y_test,verbose=1)
    y_pred = model2.predict(x_test)
    return history,y_pred

def model3(c,x_tr,x_val,y_tr,y_val,x_test,y_test):

    model3 = Sequential()
    model3.add(Input((c,)))
    model3.add(Dense(200,activation='sigmoid'))
    model3.add(Dense(1))
    model3.compile(loss='mse',optimizer='adam',metrics=['mse'])
    history = model3.fit(x_tr,y_tr,epochs=200,validation_data=(x_val,y_val),verbose=1)
    model3.evaluate(x_test,y_test,verbose=1)
    y_pred = model3.predict(x_test)
    return history,y_pred

def plot(history,y_test,y_pred):

    plt.plot(history.history['mse'],label="Training")
    plt.plot(history.history['val_mse'],label="Validation")
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Test Set Predictions")
    plt.show()


def main():

    cpu = fetch_openml(data_id=42369)
    x_tr, x_val, x_test, y_tr, y_val, y_test = preprocess_data(cpu)
    test_errors = []
    history1, pred1 = model1(9, x_tr, x_val, y_tr, y_val, x_test, y_test)
    test_errors.append(mean_squared_error(y_test, pred1))
    history2, pred2 = model2(9, x_tr, x_val, y_tr, y_val, x_test, y_test)
    test_errors.append(mean_squared_error(y_test, pred2))
    history3, pred3 = model3(9, x_tr, x_val, y_tr, y_val, x_test, y_test)
    test_errors.append(mean_squared_error(y_test, pred3))

    models = ["Model 1 (5 nodes)", "Model 2 (30 nodes)", "Model 3 (200 nodes)"]
    results_df = pd.DataFrame({"Model": models, "Test MSE": test_errors})
    print(results_df)

    if history1:
        plot(history1,y_test,pred1)
    if history2:
        plot(history2,y_test,pred1)
    if history3:
        plot(history3,y_test,pred1)

def main2():

    cpu2 = fetch_openml(data_id=503)
    x_tr, x_val, x_test, y_tr, y_val, y_test = preprocess_data(cpu2)
    test_errors = []
    history1, pred1 = model1(14, x_tr, x_val, y_tr, y_val, x_test, y_test)
    test_errors.append(mean_squared_error(y_test, pred1))
    history2, pred2 = model2(14, x_tr, x_val, y_tr, y_val, x_test, y_test)
    test_errors.append(mean_squared_error(y_test, pred2))
    history3, pred3 = model3(14, x_tr, x_val, y_tr, y_val, x_test, y_test)
    test_errors.append(mean_squared_error(y_test, pred3))

    models = ["Model 1 (5 nodes)", "Model 2 (30 nodes)", "Model 3 (200 nodes)"]
    results_df = pd.DataFrame({"Model": models, "Test MSE": test_errors})
    print(results_df)

    if history1:
        plot(history1,y_test,pred1)
    if history2:
        plot(history2,y_test,pred1)
    if history3:
        plot(history3,y_test,pred1)

main()
main2()
