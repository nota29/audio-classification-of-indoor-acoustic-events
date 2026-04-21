import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn import preprocessing


def confmatrixplot(test_df, y_pred, Filepath):
    """ A simple function plotting and saving a model's confusion matrix

        :parameter test_df: Dataframe of test samples
        :parameter y_pred: The predictions made using trained model and test samples
        :parameter Filepath: Filepath to save figure
    """
    print("Calculating confusion matrix")
    labels = test_df['Label']
    char_labels = np.unique(labels)
    print(char_labels)
    # encode string train_labels to numerical ones
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    print(classification_report(y_pred=y_pred, y_true=labels, target_names=list(char_labels), digits=3))
    # plotting the confusion matrix
    M = confusion_matrix(y_true=labels, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(15, 10))
    ConfusionMatrixDisplay(M, display_labels=char_labels).plot(xticks_rotation=-30, ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig(Filepath)
    plt.show()


def metric_plot(hist, Filepath):
    """A simple function creating plots of accuracy and loss during the training of the model

        :parameter hist: the training history of the model, containing traning and validation accuracy and loss for each epoch the model trained
        :parameter Filepath: Filepath to save figure
    """
    fig2, axs = plt.subplots(1, 2, figsize=(18, 10))
    fig2.suptitle('Accuracy & Loss Plots', fontsize=13)
    # Plot Metrics
    axs[0].plot(hist.history['accuracy'])
    axs[0].plot(hist.history['val_accuracy'])
    axs[0].set_title('Training-Validation Accuracy', fontsize=12)
    axs[1].plot(hist.history['loss'])
    axs[1].plot(hist.history['val_loss'])
    axs[1].set_title('Training-Validation Loss', fontsize=12)
    fig2.legend(['Training Set', 'Validation Set'], loc='lower right', fontsize=15)
    plt.savefig(Filepath)
    plt.show()


def report_f1score(test_df, y_pred):
    labels = test_df['Label']
    char_labels = np.unique(labels)
    print(char_labels)
    # encode string train_labels to numerical ones
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    f1score = f1_score(y_true=labels, y_pred=y_pred, average='macro')
    return f1score
