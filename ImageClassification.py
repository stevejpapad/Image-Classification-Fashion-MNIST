import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model

import Visualisations
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, array_to_img
from keras.applications import VGG16, VGG19, ResNet50
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Input
from Utilities import model_eval, data_preparation

# 'Global' Reduce Learning and Early Stopping for all the models
reduce_learning = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=2,
    min_lr=0,
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=7,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

callbacks = [reduce_learning, early_stopping]


# Baseline Model
# A simple Multi-layer perceptron, taken from Tensorflow's tutorial
def baseline_model(depth):
    x_train, y_train, x_test, y_test, x_valid, y_valid = data_preparation()

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    if depth > 1:
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()
    history = model.fit(x_train, y_train,
                        epochs=30,
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks
                        )

    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
    print('\nTrain accuracy:', train_acc)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc, ' \n')

    # Training history plotting
    Visualisations.plot_history(history, 'Baseline')

    # Model evaluation for prediction performance on the Test Set
    model_eval(model, x_test, y_test)

    # model.save('saved/Baseline NN Model')

    # Prints the architectural structure of the model
    # visualisations.visualise_predictions(model, X_test, y_test, 5, 5)


# Convolutional Neural Network with the option of {1 to 3} CV layers
def conv_nn(conv_depth):
    x_train, y_train, x_test, y_test, x_valid, y_valid = data_preparation()

    # Reshape input data from (28, 28) to (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_valid = to_categorical(y_valid, 10)
    y_test = to_categorical(y_test, 10)

    # Print training set shape
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

    # Print the number of training, validation, and test datasets
    print(x_train.shape[0], 'train set')
    print(x_valid.shape[0], 'validation set')
    print(x_test.shape[0], 'test set')

    # Model Definition
    # One hidden CONV layer
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    # Add a second hidden layer
    if conv_depth > 1:
        model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))

    # Add a third hidden layer
    if conv_depth > 2:
        model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
        model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train,
                        y_train,
                        batch_size=64,
                        epochs=30,
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks,
                        verbose=1
                        )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Training history plotting
    Visualisations.plot_history(history, 'CNN')

    # From one-hot-encoded representation -> back to categorical values
    rounded_labels = np.argmax(y_test, axis=1)

    # Model evaluation for prediction performance on the Test Set
    model_eval(model, x_test, rounded_labels)

    # model.save('saved/C-NN Model')
    plot_model(model, to_file='CNN_plot.png', show_shapes=True, show_layer_names=True)


# Transfer Learning from Pre-trained VGG model. Choice between VGG16 and VGG19
def vgg_cnn(version):
    x_train, y_train, x_test, y_test, x_valid, y_valid = data_preparation()

    x_train = np.stack((x_train,) * 3, axis=-1)
    x_valid = np.stack((x_valid,) * 3, axis=-1)
    x_test = np.stack((x_test,) * 3, axis=-1)

    # Each model (VGG16 and VGG19) require a different input size of 48*48 and 150*150 respectively
    if version == 16:
        size = 48
    elif version == 19:
        size = 150

    train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((size, size))) for im in x_train])
    valid_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((size, size))) for im in x_valid])
    test_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((size, size))) for im in x_test])

    train_Y = to_categorical(y_train)
    valid_Y = to_categorical(y_valid)
    test_Y = to_categorical(y_test)

    # for VGG of 16 layers
    if version == 16:
        from keras.applications.vgg16 import preprocess_input
        train_X = preprocess_input(train_X)
        valid_X = preprocess_input(valid_X)
        test_X = preprocess_input(test_X)

        vgg = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(48, 48, 3))
        dimension = 1

    # for VGG of 19 layers
    elif version == 19:
        from keras.applications.vgg19 import preprocess_input
        train_X = preprocess_input(train_X)
        valid_X = preprocess_input(valid_X)
        test_X = preprocess_input(test_X)
        vgg = VGG19(weights='imagenet',
                    include_top=False,
                    input_shape=(150, 150, 3),
                    classes=10)
        dimension = 4

    # Transfer learning from pre-trained VGG model on the imagenet dataset
    train_features = vgg.predict(np.array(train_X), batch_size=256, verbose=1)
    test_features = vgg.predict(np.array(test_X), batch_size=256, verbose=1)
    val_features = vgg.predict(np.array(valid_X), batch_size=256, verbose=1)
    print(train_features.shape, "\n", test_features.shape, "\n", val_features.shape)
    train_features_flat = np.reshape(train_features, (train_features.shape[0], dimension * dimension * 512))
    test_features_flat = np.reshape(test_features, (test_features.shape[0], dimension * dimension * 512))
    val_features_flat = np.reshape(val_features, (val_features.shape[0], dimension * dimension * 512))

    # Model Definition
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=(dimension * dimension * 512)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    history = model.fit(
        train_features_flat,
        train_Y,
        epochs=50,
        validation_data=(val_features_flat, valid_Y),
        callbacks=callbacks)

    # Training history plotting
    model_name = 'VGG'+str(version)
    Visualisations.plot_history(history, model_name)

    # From one-hot-encoded representation -> back to categorical values
    rounded_labels = np.argmax(test_Y, axis=1)

    # Model evaluation for prediction performance on the Test Set
    model_eval(model, test_features_flat, rounded_labels)

    # Visualisations.visualise_predictions(model, x_test, y_test, 5, 5)
    model.save("saved/VGG NN Model")

# Convolutional Neural networks with the addition of 'Image Augmentation'
# Choice Between 1, 2 or 3 layers of CNNs
def image_augmentation(conv_depth):
    x_train, y_train, x_test, y_test, x_valid, y_valid = data_preparation()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_valid = to_categorical(y_valid, 10)
    y_test = to_categorical(y_test, 10)

    # Model Definition
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    if conv_depth > 1:
        model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))

    if conv_depth > 2:
        model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
        model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    gen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.08,
                             # rotation_range=10,
                             # fill_mode='constant',
                             # horizontal_flip=False,
                             # vertical_flip=False,
                             # shear_range=0.3,
                             # horizontal_flip=True
                             )
    batches = gen.flow(x_train, y_train, batch_size=256)
    val_batches = gen.flow(x_valid, y_valid, batch_size=256)

    history = model.fit_generator(batches,
                                  steps_per_epoch=x_train.shape[0] // 256,
                                  epochs=50,
                                  validation_data=val_batches,
                                  validation_steps=x_valid.shape[0] // 256,
                                  callbacks=callbacks)

    # Training history plotting
    Visualisations.plot_history(history, 'Image Augmentation')

    # From one-hot-encoded representation -> back to categorical values
    rounded_labels = np.argmax(y_test, axis=1)

    # Model evaluation for prediction performance on the Test Set
    model_eval(model, x_test, rounded_labels)

    # model.save("saved/Augmentation NN Model")
    # plot_model(model, to_file='augmentation_plot.png', show_shapes=True, show_layer_names=True)
