# Import required libraries for performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
# Import required libraries for machine learning classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Import others
import pandas as pd
from sklearn.metrics import confusion_matrix
from warnings import simplefilter
from sklearn.metrics import classification_report
# For RNN model
import numpy as np
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils.vis_utils import plot_model
import seaborn as sns


def plot_cm(matrix, color):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix / np.sum(matrix), fmt='.2%', annot=True, cmap=color)
    ax.set_title(f'Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['A', 'D', 'H'])
    ax.yaxis.set_ticklabels(['A', 'D', 'H'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Define dictionary with performance metrics
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'f1_score': make_scorer(f1_score, average='macro')}

# Instantiate the machine learning classifiers
svc_model = SVC(coef0=5, kernel='poly')
rfc_model = RandomForestClassifier()
kfolds = KFold(n_splits=10)


# Define the models evaluation function
def models_evaluation_training(X, y):
    X = X
    y = y
    y_true = y
    estimator = keras_estimator(X, y)
    # Perform cross-validation to each machine learning classifier
    svc_val = cross_validate(svc_model, X, y, cv=kfolds, scoring=scoring)
    rfc_val = cross_validate(rfc_model, X, y, cv=kfolds, scoring=scoring)
    mlp_val = mlp_function(X, y)
    # Perform cross-validation prediction to each machine learning models
    svc_predict = cross_val_predict(svc_model, X, y, cv=kfolds)
    rfc_predict = cross_val_predict(rfc_model, X, y, cv=kfolds)
    mlp_predict = cross_val_predict(estimator, X, y, cv=kfolds)
    print("SVC prediction confusion matrix")
    print(confusion_matrix(y, svc_predict, labels=["A", "D", "H"]))
    plot_cm(confusion_matrix(y, svc_predict, labels=["A", "D", "H"]), 'Reds')
    plt.show()
    print("RFC prediction confusion matrix")
    print(confusion_matrix(y, rfc_predict, labels=["A", "D", "H"]))
    plot_cm(confusion_matrix(y, rfc_predict, labels=["A", "D", "H"]), 'Blues')
    plt.show()
    print("MLP prediction confusion matrix")
    print(confusion_matrix(y, mlp_predict, labels=["A", "D", "H"]))
    plot_cm(confusion_matrix(y, mlp_predict, labels=["A", "D", "H"]), 'Greens')
    plt.show()

    print("SVC prediction classification report")
    print(classification_report(list(y), svc_predict))
    print("RF prediction classification report")
    print(classification_report(list(y), rfc_predict))
    print("MLP prediction classification report")
    print(classification_report(list(y), mlp_predict))

    print("Summary table")
    table = []
    table = all_model_score_table(svc_val, rfc_val, mlp_val)
    return table


#
def all_model_score_table(svc, rfc, mlp):
    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({f'SVC': [svc['test_accuracy'].mean(),
                                                 svc['test_precision'].mean(),
                                                 svc['test_recall'].mean(),
                                                 svc['test_f1_score'].mean()],

                                        f'RF': [rfc['test_accuracy'].mean(),
                                                rfc['test_precision'].mean(),
                                                rfc['test_recall'].mean(),
                                                rfc['test_f1_score'].mean()],

                                        f'MLP': [mlp['test_accuracy'].mean(),
                                                 mlp['test_precision'].mean(),
                                                 mlp['test_recall'].mean(),
                                                 mlp['test_f1_score'].mean()]
                                        },

                                       index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    # Return models performance metrics scores data frame
    return (models_scores_table)


def keras_estimator(X, y):
    estimator = KerasClassifier(build_fn=baseline_model, number_of_features=X.shape[1],
                                epochs=100, batch_size=5, verbose=0)
    return estimator


def mlp_function(X, y):
    # evaluate the keras model
    estimator = keras_estimator(X, y)
    results = cross_validate(estimator, X, y, cv=kfolds, scoring=scoring)
    return results


def summary_mlp():
    keras_model = baseline_model()
    print(keras_model.summary())


def baseline_model(number_of_features=17):
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=number_of_features, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

# Prediction models refer to different websites:
# https://towardsdatascience.com/complete-guide-to-pythons-cross-validation-with-examples-a9676b5cac12
# https://stackoverflow.com/questions/43275506/scikit-learn-cross-val-predict-only-works-for-partitions
# https://zhuanlan.zhihu.com/p/38200980
# https://github.com/motapinto/football-classification-predications/blob/master/src/Supervised%20Learning%20Models.ipynb
# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
# https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
# https://medium.com/nerd-for-tech/premier-league-predictions-using-artificial-intelligence-7421dddc8778
# https://blog.csdn.net/weixin_44551646/article/details/112911215
