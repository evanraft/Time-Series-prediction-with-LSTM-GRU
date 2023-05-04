# Let`s import all packages that we may need:

import sys
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.model_selection import train_test_split # to split the data into two parts
#from sklearn.cross_validation import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras_tuner
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# !pip install -q yfinance
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout

import tensorflow as tf

x_train=[]
def import_dataset(dataset_name):
    import pandas as pd  #
    if dataset_name == 'stock':
        # STOCK PICES


        # For reading stock data from yahoo
        # from pandas_datareader.data import DataReader
        import yfinance as yf

        # from pandas_datareader import data as pdr

        yf.pdr_override()

        # For time stamps
        from datetime import datetime
        comp='GOOG'
        # The tech stocks we'll use for this analysis
        tech_list = [comp]

        # Set up End and Start times for data grab
        tech_list = [comp, ]

        end = datetime.now()
        start = datetime(end.year - 10, end.month, end.day)

        for stock in tech_list:
            globals()[stock] = yf.download(stock, start, end)

        company_list = [GOOG]
        company_name = [comp]

        for company, com_name in zip(company_list, company_name):
            company["company_name"] = com_name

        df = pd.concat(company_list, axis=0)
        df.tail(10)

        df = df.drop('company_name', axis=1)
        col_to_move = 'Close'
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index(col_to_move)))
        df = df[cols]
        #df = df[['Close']]
        return df

    if dataset_name == 'electric':

        # #### Household Electric Power Consumption #####
        # Data can be downloaded from: http://archive.ics.uci.edu/ml/machine-learning-databases/00235/
        # Just open the zip file and grab the file 'household_power_consumption.txt' put it in the directory
        # that you would like to run the code.


        df = pd.read_csv('household_power_consumption.txt', sep=';',
                         parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
                         low_memory=False, na_values=['nan','?'], index_col='dt')
        ## finding all columns that have nan:

        droping_list_all = []
        for j in range(0, 7):
            if not df.iloc[:, j].notnull().all():
                droping_list_all.append(j)
                # print(df.iloc[:,j].unique())
        print(droping_list_all)

        # filling nan with mean in any columns

        for j in range(0, 7):
            df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())

        # another sanity check to make sure that there are not more any nan
        df.isnull().sum()

        # resampling of data over hour
        df = df.resample('h').mean()
        print (df.shape)
        #df.head()
        return df

    if dataset_name == 'jena_climate':
        ##### jena_climate_2009_2016 #####

        df = pd.read_csv('jena_climate_2009_2016.csv', sep=',', low_memory=False, na_values=['nan','?'])
        col_to_move = 'T (degC)'
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index(col_to_move)))
        df = df[cols]
        df = df.drop('Date Time', axis=1)
        # Remove the last 100 rows
        df = df[:-300000]
        return df

    if dataset_name == 'climate_delhi':
        df = pd.read_csv('DailyDelhiClimateTrain.csv', sep=',', low_memory=False, na_values=['nan', '?'])
        # col_to_move = 'T (degC)'
        # cols = list(df.columns)
        # cols.insert(0, cols.pop(cols.index(col_to_move)))
        # df = df[cols]
        # df = df.drop('date', axis=1)
        # df = df["meantemp"]
        # df.head()
        # # Remove the last 100 rows
        # df = df.drop('p (mbar)', axis=1)
        # df = df.drop('rh (%)', axis=1)
        # df = df[:-300000]
        # #df = df.loc[:, ['T (degC)']] # using loc

        n_cols = 1
        df = df[['meantemp']]
        return df

    if dataset_name == 'Electric_Production':
        df = pd.read_csv('Electric_Production.csv', sep=',', low_memory=False, na_values=['nan', '?'])

        n_cols = 1
        df = df.drop('DATE', axis=1)
        return df

### GENERAL


def create_train_val(window_size, future, df):
    # Create a new dataframe with only the 'Close column
    data = df
    # Convert the dataframe to a numpy array
    dataset = data.values
    #print (dataset)
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .60 ))

    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    #print (scaled_data)

    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(window_size, len(train_data) - future):
        x_train.append(train_data[i - window_size:i, :])
        # y_train.append(train_data[i, 0])
        y_train.append(train_data[i + future, 0])  # use i+72 instead of the next value
    #     if i<= 61:
    #         print(x_train)
    #         print(y_train)
    #         print()

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #print(x_train.shape)
    #print(y_train.shape)
    # Reshape the data
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], num_columns))
    #print(x_train.shape)
    # input_shape = x_train.shape[1:]
    # print (input_shape)
    # x_train = x_train[:, 0, :]
    # print(x_train.shape)

    # # Create the scaled validation data set
    # val_data_len = int(np.ceil(len(dataset) * .10 ))
    # valid_data = scaled_data[int(training_data_len):int(training_data_len)+int(val_data_len), :]
    # #print (val_data_len)
    # # Split the data into x_valid and y_valid data sets
    # x_val = []
    # y_val = []
    #
    # for i in range(window_size, len(valid_data)-future):
    #     x_val.append(valid_data[i-window_size:i, :])
    #     y_val.append(valid_data[i+future, 0])
    #
    # # Convert the x_valid and y_valid to numpy arrays
    # x_val, y_val = np.array(x_val), np.array(y_val)
    # #print (x_val.shape)
    #
    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002
    #test_data = scaled_data[int(training_data_len) + int(val_data_len):, :] ########
    test_data = scaled_data[int(training_data_len):, :]
    print('x_train: ', x_train.shape)
    print('y_train: ', y_train.shape)
    # print('x_val: ', x_val.shape)
    # print('y_val: ', y_val.shape)

    # Create the data sets x_test and y_test
    x_test = []
    y_test = []
    # y_test = dataset[int(training_data_len)+int(val_data_len):, 0]
    # print (y_test)
    for i in range(window_size, len(test_data) - future):
        x_test.append(test_data[i - window_size:i, :])
        y_test.append(test_data[i + future, 0])
    # Convert the data to a numpy array
    x_test = np.array(x_test)
    # x_test, y_test = np.array(x_test), np.array(y_test)
    #print(x_test.shape)
    # print (y_test)
    # Reshape the data
    print (x_test.shape[0], x_test.shape[1], x_train.shape[2])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_train.shape[2]))

    print ('x_test: ', x_test.shape)
    #print ('y_test: ', y_test.shape)
    #return train_data, x_train, y_train, x_val, y_val, x_test, y_test, scaler
    return train_data, x_train, y_train, x_test, y_test, scaler


def create_model(x_train,mod):
    if mod == 1:
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        #model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    if mod == 2:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
            LSTM(64, return_sequences=False),
            Dense(32),
            Dense(16),
            Dense(1)
        ])

        #model.compile(optimizer='adam', loss='mse', metrics="mean_absolute_error")
        return model


def build_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True, input_shape= (x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(GRU(hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=False))
    #model.add(Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(hp.Int('dense_units', min_value=16, max_value=124, step=16)))
    model.add(Dense(1))
    #model.compile(optimizer='adam', loss='mean_squared_error')
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd'], default='adam')
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

def tuner(x_train, y_train, x_val, y_val, batch_size, epochs):
    callbacks = [
        # tf.keras.callbacks.ModelCheckpoint(
        #     "corn_tranfer_" + "classes_" + str(number_of_classes) + "_Cnn_" + str(cnn) + "_loss_" + str(
        #         loss_func) + ".h5", save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)]

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        directory='my_dir',
        overwrite=True,
        project_name='my_project')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(x_train, y_train,
                 validation_data=(x_val, y_val),
                 batch_size=batch_size,
                 epochs=epochs)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    final_model = build_model(best_hp)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    best_hyperparameters_values = best_hyperparameters.values
    final_model = build_model(best_hps)
    best_model_config = final_model.get_config()
    # Print all the hyperparameters used to create the best model
    for key in best_hyperparameters_values.keys():
        print(f"{key}: {best_hyperparameters_values[key]}")


    history = final_model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[stop_early])
    # print("test loss", results[0])
    # print("test acc:", results[1])
    return history, final_model

def mae_loss(y_pred, y_true):
    abs_error = np.abs(y_pred - y_true)
    sum_abs_error = np.sum(abs_error)
    loss = sum_abs_error / y_true.size
    return loss

def predictions(x_test, model, train_data, scaler, y_test, dataset_name, batch_size, epochs, window_size, tuner):
    predictions = model.predict(x_test)
    predictions = np.reshape(predictions, (predictions.shape[0], 1))
    # print (predictions)
    predictions = np.hstack((predictions, np.zeros((predictions.shape[0], train_data.shape[1] - 1))))
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions[:, 0]
    # print (predictions[:-5])

    y_test = np.array(y_test)
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    y_test = np.hstack((y_test, np.zeros((y_test.shape[0], train_data.shape[1] - 1))))
    y_test = scaler.inverse_transform(y_test)
    y_test = y_test[:, 0]
    print(len(y_test))
    # print(y_test[:-5])
    # Get the root mean squared error (RMSE)
    mse = np.mean(((predictions - y_test) ** 2))
    print("MSE = ", mse)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print("RMSE = ", rmse)

    mae = mae_loss(predictions, y_test)
    print('MAE: ', mae)
    RMSE = np.sqrt(np.mean((predictions - y_test) ** 2))
    # print (RMSE)

    preds_acts = pd.DataFrame(data={'Predictions': predictions, 'Actuals': y_test})
    # preds_acts

    plt.figure(figsize=(16, 6))
    plt.plot(preds_acts['Predictions'][:500])
    plt.plot(preds_acts['Actuals'][:500])
    plt.legend(['Prediction', 'Actuals'])
    #plt.show()
    if tuner == False:
        plt.savefig('plot.png')
    if tuner == True:
        plt.savefig('plot_tuner.png')
    results_df = pd.DataFrame(columns=['info', 'predictions', 'actual', 'rmse', 'mae', 'mse'])
    info = f'dataset={dataset_name}_bs={batch_size}_epochs={epochs}.csv'
    res = pd.DataFrame({
        'info': info,
        'predictions': predictions ,
        'actual': y_test ,
        'rmse': rmse,
        'mae': mae,
        'mse': mse
    })
    print (type(predictions))
    results_df = results_df.append(res)
    results_df = results_df.append(pd.Series(), ignore_index=True)
    results_df.to_csv(f'dataset_{dataset_name}_GOOG_bs_{batch_size}_epochs_{epochs}_win_size_{window_size}_col_{x_test.shape[2]}=.csv', decimal=',',  sep=';', index=False)

    return mse, rmse, mae, predictions

def main():
    global x_train
    batch_size = 1
    epochs = 15

    # df = pd.read_csv('jena_climate_2009_2016.csv', sep=',', low_memory=False, na_values=['nan', '?'])
    # print (df.head())
    dataset_name = 'Electric_Production'
    df = import_dataset(dataset_name)
    print (df.columns)
    num_columns = df.shape[1]
    first_col_name = df.columns[0]
    print ("number of columns: ", num_columns)
    print ('name of the prediction column: ', first_col_name)
    if dataset_name == 'stock' or dataset_name == 'Electric_Production':
        window_size = 30
        future = 0
    elif dataset_name == 'electric':
        window_size = 100
        future = 6
    elif dataset_name == 'jena_climate':
        window_size = 720
        future = 60
    elif dataset_name == 'climate_delhi':
        window_size = 60
        future = 0

    #train_data, x_train, y_train, x_val, y_val, x_test, y_test, scaler = create_train_val(window_size, future, df=df)
    train_data, x_train, y_train, x_test, y_test, scaler = create_train_val(window_size, future, df=df)
    model = create_model(x_train,mod=1)
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, batch_size, epochs, validation_split=0.1)

    mse, rmse, mae, pred = predictions(x_test, model, train_data, scaler, y_test, dataset_name, batch_size, epochs, window_size,  tuner=False)


    #history, final_model = tuner(x_train, y_train, x_val, y_val, batch_size, epochs)
    #mse, rmse, mae, pred = predictions(x_test, model, train_data, scaler, y_test, dataset_name, batch_size, epochs, tuner=True)


if __name__ == "__main__":
    main()
