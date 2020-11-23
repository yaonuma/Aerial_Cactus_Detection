import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources

TERMCOLOR = True
if TERMCOLOR:
    from termcolor import colored
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 500)
pd.set_option('display.width', 10000)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



def list_packages_versions():
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    for item in installed_packages_list:
        print(item)
    print('\n')
    return


def printm(text, color='white', switch=False):
    # wrapper for standard print function
    if switch:
        print(colored(text, color))
    else:
        print(text)
    return


def data_raw2ImageDataGenerator(path):
    label_dict = {}
    with open(path + 'train.csv', 'r') as a_file:
        for line in a_file:
            pair = line.split(',')
            if pair[1].rstrip() == '0':
                label = 'not_has_cactus'
            else:
                label = 'has_cactus'
            label_dict[pair[0]] = label
    a_file.close()

    # create the flow_from_directory folders
    if not os.path.exists(path + '/train/has_cactus'):
        os.makedirs(path + '/train/has_cactus')
    if not os.path.exists(path + '/train/not_has_cactus'):
        os.makedirs(path + '/train/not_has_cactus')
    if not os.path.exists(path + '/submissions'):
        os.makedirs(path + '/submissions')

    # go through each image in the original folder and look up its placement in the dictionary
    for image in glob.glob(path + '/train/*.jpg'):
        image = image.split('/')[-1]
        shutil.move(path + 'train/' + image, path + '/train/' + label_dict[image] + '/' + image)

    return


def build_model(data_root_path):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print(model.summary())

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0005),
                  metrics=['accuracy'])

    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2)

    train_data_dir = data_root_path + "/train"
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        subset='training',
        shuffle=True,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        subset='validation',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary')

    # model fitting
    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  epochs=20,
                                  verbose=1,
                                  shuffle=True,
                                  callbacks=callbacks)

    # summarize history for accuracy and loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
    plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()

    return model


def write_submission(data_root_path, model):
    test_folder = data_root_path + "test/"

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        directory=test_folder,
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    pred = model.predict_generator(test_generator, verbose=1)
    pred_binary = [1 if value < 0.50 else 0 for value in pred]  # polarity reversed here beacuse of internal
    # mapping to class labels. can fix this by defining the mapping explicitly

    csv_file = open("submission.csv", "w")
    csv_file.write("id,has_cactus\n")
    for filename, prediction in zip(test_generator.filenames, pred_binary):
        name = filename.split("/")[1].replace(".tif", "")
        csv_file.write(str(name) + "," + str(prediction) + "\n")
    csv_file.close()

    return pred_binary


if __name__ == '__main__':
    # look at versions to avoid confusion with deprecation/compatibility
    # list_packages_versions()

    # do some proprocessing to make it compatible with the flow_from_directory() method
    data_root_path = 'data/'
    data_raw2ImageDataGenerator(data_root_path)

    # model the training data with simple 2DConvNet from scratch
    model_cactus = build_model(data_root_path)

    # get model results write model results in the format required by the Kaggle competition.
    probabilities = write_submission(data_root_path, model_cactus)