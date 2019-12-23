import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn import linear_model
import tensorflow as tf;
from tensorflow import keras
from keras import models, layers, optimizers, regularizers, callbacks
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from sklearn.model_selection import RepeatedKFold, KFold

"""
Performs a linear regression of the data and plots the output. 

Parameters:
    filename - string location of the data
    lasso - boolean to indicate whether or not to perform lasso regression
"""
def linear_regression(filename, lasso, a):
    x_train = np.load(filename + '_x_train.npy')
    x_test = np.load(filename + '_x_test.npy')
    y_train = np.load(filename + '_y_train.npy')
    y_test = np.load(filename + '_y_test.npy')

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.fit_transform(y_test)

    # Create linear regression object
    if lasso:
        regr = linear_model.Lasso(alpha=a)
    else:
        regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_test_pred = regr.predict(x_test)

    y_train_pred = regr.predict(x_train)

    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    print (filename[8:] + "-" + str(lasso) + "-" + str(a))
    # The mean squared error
    print("Training MAE:", round(mean_absolute_error(y_train, y_train_pred),4))
    print("Validation MAE:", round(mean_absolute_error(y_test, y_test_pred),4))

    y_test_pred = y_test_pred.reshape(-1, 1)
    y_test_pred_unscaled = scaler.inverse_transform(y_test_pred)

    y_train_pred = y_train_pred.reshape(-1, 1)
    y_train_pred_unscaled = scaler.inverse_transform(y_train_pred)

    y_train = y_train.reshape(-1,1)
    y_train_unscaled = scaler.inverse_transform(y_train)

    y_test = y_test.reshape(-1,1)
    y_test_unscaled = scaler.inverse_transform(y_test)


    # Scatterplot of predicted vs. actual values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Predicted vs. actual values', fontsize=14, y=1)
    plt.subplots_adjust(top=0.93, wspace=0)

    ax1.scatter(y_test_unscaled, y_test_pred_unscaled, s=2, alpha=0.7)
    ax1.plot(list(range(6,10)), list(range(6,10)), color='black', linestyle='--')
    ax1.set_title('Test set')
    ax1.set_xlabel('Actual values')
    ax1.set_ylabel('Predicted values')

    ax2.scatter(y_train_unscaled, y_train_pred_unscaled, s=2, alpha=0.7)
    ax2.plot(list(range(6,10)), list(range(6,10)), color='black', linestyle='--')
    ax2.set_title('Train set')
    ax2.set_xlabel('Actual values')
    ax2.set_ylabel('')
    ax2.set_yticklabels(labels='')

    #plt.show()
    plt.savefig("../figs/" + filename[8:] + "-" + str(lasso) + "-" + str(a) + ".png")
    plt.close()

"""
Builds a neural network model and plots the output.

Parameters:
    filename - string location of the data
    variance_threshold - variance threshold
    l1 - lasso penalty (number)
    learning_rate - alpha size (number)
    es_patience - patience for early stopping
    epochs - number of epochs to run the model
"""
def make_model(filename, variance_threshold, l1, learning_rate, es_patience, epochs):
    x_train = np.load(filename + '_x_train.npy')
    x_test = np.load(filename + '_x_test.npy')
    y_train = np.load(filename + '_y_train.npy')
    y_test = np.load(filename + '_y_test.npy')

    selector = VarianceThreshold(threshold=variance_threshold)
    x_train = selector.fit_transform(x_train)
    feature_mask = selector.get_support()
    del_features = np.array([i if feature_mask[i] == False else -1 for i in range(feature_mask.shape[0])])
    del_features = del_features[del_features != -1]
    x_test = np.delete(x_test, del_features, axis=1)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.fit_transform(y_test)

    # BUILDING THE MODEL
    nn2 = models.Sequential()
    nn2.add(layers.Dense(128, input_shape=(x_train.shape[1],),
     activation='relu', kernel_regularizer=regularizers.l1(l=l1)))
    nn2.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l1(l=l1)))
    nn2.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l1(l=l1)))
    nn2.add(layers.Dense(1, activation='relu'))

    # Compiling the model
    nn2.compile(loss='mean_absolute_error',
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['mean_absolute_error'])

    # Printing the model summary
    # print(nn2.summary())

    # Visualising the neural network
    SVG(model_to_dot(nn2, show_layer_names=False, show_shapes=True).create(prog='dot', format='svg'))

    # Training the model
    es = callbacks.EarlyStopping(monitor='val_loss', patience=es_patience, mode='min', restore_best_weights=True)

    nn2_history = nn2.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=256,
                    validation_split = 0.1,
                    callbacks = [es],
                    verbose=0)

    if "all_words" in filename:
        dataset = "AW"
    elif "character_lines" in filename:
        dataset = "CL"
    elif "character_words" in filename:
        dataset = "CW"
    model_info = dataset + "-" + str(variance_threshold) + "-" + str(l1) + "-" + str(learning_rate)
    nn_model_evaluation(nn2, 0, x_train, x_test, y_train, y_test, scaler, model_info)


"""
Wrapper function that utilizes make_kfold_model_helper to build neural networks implementing kfold cross validation.

Parameters:
    filename - string location of the data
    variance_threshold - variance threshold
    l1 - lasso penalty (number)
    learning_rate - alpha size (number)
    es_patience - patience for early stopping
    epochs - number of epochs to run the model
"""
def make_kfold_model(filename, variance_threshold, l1, learning_rate, es_patience, epochs):
    print(filename)
    data = pd.read_csv(filename)
    IMDB_scores = np.load('../data/IMDB_scores.npy')
    IMDB_scores = IMDB_scores.reshape(-1, 1)
    n_splits = 187
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=None) 
    train_maes = []
    validation_maes = []

    #Variance threshold?

    for train_index, test_index in kf.split(data):
        x_train, x_test = data.loc[train_index, :], data.loc[test_index, :] 
        y_train, y_test = IMDB_scores[train_index], IMDB_scores[test_index]
        train_mae, validation_mae = make_kfold_model_helper(variance_threshold, l1, learning_rate, es_patience, epochs, x_train, x_test, y_train, y_test)
        train_maes.append(train_mae)
        validation_maes.append(validation_mae)

    train_maes = np.array(train_maes)
    validation_maes = np.array(validation_maes)

    print("Train MAES: ", train_maes)
    print("Validation MAES: ", validation_maes)

    print("Average Train MAE: ", train_maes.mean())
    print("Average Validation MAE: ", validation_maes.mean())


"""
Builds a neural network model on a kfold section of data. 

Parameters:
    filename - string location of the data
    variance_threshold - variance threshold
    l1 - lasso penalty (number)
    learning_rate - alpha size (number)
    es_patience - patience for early stopping
    epochs - number of epochs to run the model
    x_train - x train data
    x_test - x test data
    y_train - y train data
    y_test - y test data

Returns:
    Mean absolute error for training and testing
"""
def make_kfold_model_helper(variance_threshold, l1, learning_rate, es_patience, epochs, x_train, x_test, y_train, y_test):
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.fit_transform(y_test)

    # BUILDING THE MODEL
    nn2 = models.Sequential()
    nn2.add(layers.Dense(128, input_shape=(x_train.shape[1],),
     activation='relu', kernel_regularizer=regularizers.l1(l=l1)))
    nn2.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l1(l=l1)))
    nn2.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l1(l=l1)))
    nn2.add(layers.Dense(1, activation='relu'))

    # Compiling the model
    nn2.compile(loss='mean_absolute_error',
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['mean_absolute_error'])

    # Visualising the neural network
    #SVG(model_to_dot(nn2, show_layer_names=False, show_shapes=True).create(prog='dot', format='svg'))

    # Training the model
    es = callbacks.EarlyStopping(monitor='val_loss', patience=es_patience, mode='min', restore_best_weights=True)

    nn2_history = nn2.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=256,
                    validation_split = 0.1,
                    callbacks = [es])

    train_mae, validation_mae = kfold_model_evaluation(nn2, 0, x_train, x_test, y_train, y_test, scaler)

    return train_mae, validation_mae


"""
Evaluates a neural network using mean absolute error.

Parameters:
    model - neural network model
    x_train - x train data
    x_test - x test data
    y_train - y train data
    y_test - y test data
    scaler - min max scaler used to normalize data

Returns:
    Mean absolute error for training and testing
"""
def kfold_model_evaluation(model, x_train, x_test, y_train, y_test, scaler):
    """
    For a given neural network model that has already been fit, prints for the train and tests sets the MSE and r squared
    values, a line graph of the loss in each epoch, and a scatterplot of predicted vs. actual values with a line
    representing where predicted = actual values. Optionally, a value for skip_epoch can be provided, which skips that
    number of epochs in the line graph of losses (useful in cases where the loss in the first epoch is orders of magnitude
    larger than subsequent epochs). Training and test sets can also optionally be specified.
    """

    # MSE and r squared values
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    y_test_pred_unscaled = scaler.inverse_transform(y_test_pred)
    y_train_pred_unscaled = scaler.inverse_transform(y_train_pred)
    y_train_unscaled = scaler.inverse_transform(y_train)
    y_test_unscaled = scaler.inverse_transform(y_test)

    train_mae = round(mean_absolute_error(y_train, y_train_pred),4)
    validation_mae = round(mean_absolute_error(y_test, y_test_pred),4)

    return train_mae, validation_mae


"""
Evaluates a neural network using mean absolute error.

Parameters:
    model - neural network model
    skip_epochs - epochs to be skipped
    x_train - x train data
    x_test - x test data
    y_train - y train data
    y_test - y test data
    scaler - min max scaler used to normalize data
    model_info - string of model info to be printed

Returns:
    Mean absolute error for training and testing
"""
def nn_model_evaluation(model, skip_epochs, x_train, x_test, y_train, y_test, scaler, model_info):
    """
    For a given neural network model that has already been fit, prints for the train and tests sets the MSE and r squared
    values, a line graph of the loss in each epoch, and a scatterplot of predicted vs. actual values with a line
    representing where predicted = actual values. Optionally, a value for skip_epoch can be provided, which skips that
    number of epochs in the line graph of losses (useful in cases where the loss in the first epoch is orders of magnitude
    larger than subsequent epochs). Training and test sets can also optionally be specified.
    """

    # MSE and r squared values
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    y_test_pred_unscaled = scaler.inverse_transform(y_test_pred)
    y_train_pred_unscaled = scaler.inverse_transform(y_train_pred)
    y_train_unscaled = scaler.inverse_transform(y_train)
    y_test_unscaled = scaler.inverse_transform(y_test)

    print (model_info)
    print("Training MAE:", round(mean_absolute_error(y_train, y_train_pred),4))
    print("Validation MAE:", round(mean_absolute_error(y_test, y_test_pred),4))
    print("\nTraining r2:", round(r2_score(y_train_unscaled, y_train_pred_unscaled),4))
    print("Validation r2:", round(r2_score(y_test_unscaled, y_test_pred_unscaled),4))
    print("--------------------------------------------------")

    # Line graph of losses
    model_results = model.history.history
    plt.plot(list(range((skip_epochs+1),len(model_results['loss'])+1)), model_results['loss'][skip_epochs:], label='Train')
    plt.plot(list(range((skip_epochs+1),len(model_results['val_loss'])+1)), model_results['val_loss'][skip_epochs:], label='Test', color='green')
    plt.legend()
    plt.title('Training and test loss at each epoch', fontsize=14)
    #plt.show()

    # Scatterplot of predicted vs. actual values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Predicted vs. actual values', fontsize=14, y=1)
    plt.subplots_adjust(top=0.93, wspace=0)

    ax1.scatter(y_test_unscaled, y_test_pred_unscaled, s=2, alpha=0.7)
    ax1.plot(list(range(6,10)), list(range(6,10)), color='black', linestyle='--')
    ax1.set_title('Test set')
    ax1.set_xlabel('Actual values')
    ax1.set_ylabel('Predicted values')

    ax2.scatter(y_train_unscaled, y_train_pred_unscaled, s=2, alpha=0.7)
    ax2.plot(list(range(6,10)), list(range(6,10)), color='black', linestyle='--')
    ax2.set_title('Train set')
    ax2.set_xlabel('Actual values')
    ax2.set_ylabel('')
    ax2.set_yticklabels(labels='')
    plt.savefig('../figs/' + model_info + ".png")
    #plt.show()

    train_mae = round(mean_absolute_error(y_train, y_train_pred),4)
    validation_mae = round(mean_absolute_error(y_test, y_test_pred),4)

    return train_mae, validation_mae

def main():
    # parameters = filename, variance_threshold, l1 value, learning_rate, es_patience, epochs
    #make_model('../data/all_words', 300, 0.002, 0.001, 50, 3000)

    """
    filename = '../data/all_words'
    es_patience = 50
    epochs = 5000
    for variance_threshold in range(0, 500, 100):
        for l1 in np.arange(0, 0.005, 0.001):
            for learning_rate in np.arange(0, 0.005, 0.001):
                make_model(filename, variance_threshold, l1, learning_rate, es_patience, epochs)
    """
    # variance_threshold = 0
    # es_patience = 50
    # epochs = 5000
    # filenames = ['../data/character_lines', '../data/character_words']
    # for filename in filenames:
    #     for l1 in np.arange(0, 0.005, 0.001):
    #             for learning_rate in np.arange(0, 0.0005, 0.0001):
    #                 make_model(filename, variance_threshold, l1, learning_rate, es_patience, epochs)
    # #make_model('../data/character_lines', 0, 0.001, 0.001, 50, 3000)
    # alphas = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
    # filenames = ['../data/all_words', '../data/character_lines', '../data/character_words',
    # '../data/character_lines_twss', '../data/character_words_twss']
    # make_kfold_model('../data/character_lines.csv', 0, 0.001, 0.001, 50, 3000)
    # a = " "
    # for filename in filenames:
    #     for alpha in alphas:
    #         linear_regression(filename, True, alpha)
    #     linear_regression(filename, False, a)
    linear_regression('../data/character_lines', True, 0.003)


if __name__ == "__main__":
    main()
