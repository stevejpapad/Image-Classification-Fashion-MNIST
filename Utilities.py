import sklearn
import keras
import numpy as np
import tensorflow as tf
from sklearn import metrics


def data_preparation():
    # Fetch the Fashion - MNIST dataset
    (img_train, labels_train), (img_test, labels_test) = keras.datasets.fashion_mnist.load_data()

    # Re-scale image data. From initial Range {0 - 255} to {0 - 1}
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    img_train = scaler.fit_transform(img_train.reshape(-1, img_train.shape[-1])).reshape(img_train.shape)
    img_test = scaler.transform(img_test.reshape(-1, img_test.shape[-1])).reshape(img_test.shape)

    # Data Summarization
    print('Train: X=%s, y=%s' % (img_train.shape, labels_train.shape))
    print('Test: X=%s, y=%s' % (img_test.shape, labels_test.shape))

    unique_elements, counts_elements = np.unique(labels_train, return_counts=True)
    print('Classes: ', unique_elements, ' counts : ', counts_elements)

    (img_train, labels_train), (img_valid, labels_valid) = split_val(img_train, labels_train)

    return img_train, labels_train, img_test, labels_test, img_valid, labels_valid


def split_val(x_train, y_train):
    # Splitting the initial test set into Validation and Test set
    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]

    # TEMPORAL! FOR CODE DEBUGGING PURPOSES ONLY!
    # (x_train, x_valid) = x_train[59000:], x_train[:100]
    # (y_train, y_valid) = y_train[59000:], y_train[:100]
    return (x_train, y_train), (x_valid, y_valid)


def model_eval(model, img_test, labels_test):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    y_predict = probability_model.predict(img_test)
    y_pred = np.argmax(y_predict, axis=1)  # most probable class prediction
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("Mean Accuracy Score: %.4f" % (metrics.accuracy_score(labels_test, y_pred) * 100))
    print("Mean Precision: %.4f" % (metrics.precision_score(labels_test, y_pred, average="macro") * 100))
    print("Mean Recall: %.4f" % (metrics.recall_score(labels_test, y_pred, average="macro") * 100))
    print("Mean F1: %.4f" % (metrics.f1_score(labels_test, y_pred, average="macro") * 100))
    # print(metrics.confusion_matrix(labels_test, y_pred))
    print(metrics.classification_report(labels_test, y_pred, target_names=class_names))