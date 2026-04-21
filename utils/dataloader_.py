from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator

def load_data(datagen, train_df, test_df, val_split, batchsize, mode=None):
    """
    Custom function to load the datasets
    Args:
    :param datagen: ImageDataGenerator object from keras
    :param train_df: a list containing the paths to the dataframes used
    :param test_df: the path of the test dataset csv
    :param val_split: percentage to split the dataset
    :param batchsize: mini batch size
    :param mode: Mode to be used, for training or evaluation purposes

    :return:
    train_iterator: A keras iterator to load images to the model with batches, for training purposes
    val_iterator: A keras iterator to load images to the model with batches, for validation purposes
    train_data_size: Number of training images
    val_data_size: Number of validation images
    test_iterator: A keras iterator to load images to the model with batches, for evaluation purposes
    test_data_size: Number of evaluation images
    """
    if mode == 'training':
        # shuffle samples
        sample_mfcc = plt.imread(train_df['Filepath'][0])
        # Find Input Shape
        x_size_mfcc, y_size_mfcc = np.shape(sample_mfcc)
        print("Load Data")
        train_iterator = datagen.flow_from_dataframe(dataframe=train_df, x_col='Filepath',
                                                     y_col='Label', target_size=(x_size_mfcc, y_size_mfcc),
                                                     class_mode='categorical', batch_size=batchsize,
                                                     color_mode='grayscale', shuffle=True, subset='training',
                                                     seed=7)
        val_iterator = datagen.flow_from_dataframe(dataframe=train_df, x_col='Filepath',
                                                   y_col='Label', target_size=(x_size_mfcc, y_size_mfcc),
                                                   class_mode='categorical', batch_size=batchsize,
                                                   color_mode='grayscale', shuffle=True, subset='validation',
                                                   seed=7)
        val_data_size = round(val_split * len(train_df))
        train_data_size = len(train_df) - val_data_size
        return train_iterator, val_iterator, train_data_size, val_data_size
    elif mode == 'test':
        sample_mfcc = plt.imread(test_df['Filepath'][0])
        # Find Input Shape
        x_size_mfcc, y_size_mfcc = np.shape(sample_mfcc)
        test_iterator = datagen.flow_from_dataframe(dataframe=test_df, x_col='Filepath', y_col='Label',
                                                    target_size=(x_size_mfcc, y_size_mfcc),
                                                    class_mode='categorical',
                                                    batch_size=batchsize, color_mode='grayscale',
                                                    shuffle=False, seed=7)
        test_data_size = len(test_df)
        return test_iterator, test_data_size



def class_weight_calc(train_df, class_mode='sklearn'):
    """
    A cumulative function that returns each classes' weights
    :param train_df: dataframe to extract train_labels from
    :param class_mode: Default = 'sklearn', computes class weights using sklearn utility
                       If class_mode == alphas, class weights are computed based on probabilies
                       If class_mode == custom, class weights are computed based on the inverse
                       of the probability of each class, then normalized by min weight
    :return: A dictionary containing the weights for each class
    """
    counts = train_df['Label'].value_counts()
    labels = train_df['Label']
    if class_mode == 'sklearn':
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = dict(enumerate(class_weights.flatten(), 0))
        return class_weights
    elif class_mode == 'alphas':
        counts = counts.sort_index()
        alphas = []
        for i in range(len(counts)):
            alpha = counts[i] / len(labels)
            alphas.append(alpha)
        for i in range(1, 7):
            alphas[i] = 1 - alphas[i]
        mydict = {}
        for i in range(len(alphas)):
            mydict[i] = alphas[i]
        return mydict
    elif class_mode == 'custom':
        class_weights2 = {}
        weight = []
        for i in range(len(counts)):
            weight.append(len(counts) / (counts[i]))
        min_weights = min(weight)
        for i in range(len(counts)):
            class_weights2[i] = weight[i] / min_weights
        return class_weights2
