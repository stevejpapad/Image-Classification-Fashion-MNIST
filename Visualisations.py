import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def sample_plot(img_train, img_label, size):
    plt.figure(figsize=(10, 10))
    for i in range(size):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[img_label[i]])
    plt.show()


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def visualise_predictions(model, img_test, labels_test, num_rows, num_cols):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(img_test)

    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], labels_test, img_test)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], labels_test)
    plt.tight_layout()
    plt.show()


def plot_history(history, model_name):
    df = pd.DataFrame(history.history)
    df = df.drop('lr', axis=1)
    df.plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(model_name + ' Model : Training History')
    plt.xlabel('Epochs')
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.show()
