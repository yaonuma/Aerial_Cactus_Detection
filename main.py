import glob
import os
import shutil

import pandas as pd
import pkg_resources
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TERMCOLOR = True
if TERMCOLOR:
    from termcolor import colored
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 500)
pd.set_option('display.width', 10000)


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
    # create a dictionary of image labels. it is best to cache this information now to avoid an N^2 image sorting
    # algorithm. Now it is 2*N. Much better
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
    DESIRED_ACCURACY = 0.99

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') >= DESIRED_ACCURACY:  # this is the stopping criterion for the training
                print("\nReached " + str(DESIRED_ACCURACY * 100) + "% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # # The fourth convolution
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # # The fifth convolution
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])

    # This code block should create an instance of an ImageDataGenerator called train_datagen
    # And a train_generator by calling train_datagen.flow_from_directory
    train_datagen = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       rescale=1 / 255,
                                       validation_split=0.2)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(data_root_path + 'train',
                                                        subset='training',
                                                        target_size=(32, 32),
                                                        batch_size=128,
                                                        class_mode='binary')
    validation_generator = train_datagen.flow_from_directory(data_root_path + 'train',
                                                             subset='validation',
                                                             target_size=(32, 32),
                                                             batch_size=128,
                                                             class_mode='binary')

    # model fitting
    history = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=20,
        steps_per_epoch=20,
        epochs=100,
        verbose=1,
        callbacks=[callbacks]
    )

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


def test_model(data_root_path, model):
    test_datagen = ImageDataGenerator(featurewise_center=True,
                                      featurewise_std_normalization=True,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      rescale=1 / 255)

    test_generator = test_datagen.flow_from_directory(data_root_path + '/test/',
                                                      target_size=(32, 32),
                                                      batch_size=16,
                                                      class_mode=None,  # only data, no labels
                                                      shuffle=False)

    probabilities = model.predict_generator(test_generator)

    return probabilities


def write_submission(raw_data_path, probs):
    test_result_dict = {}

    for i, image in enumerate(glob.glob(raw_data_path + 'test/no_label/*.jpg')):
        image = image.split('/')[-1]
        test_result_dict[image] = probs[i][0]

    df_submission = pd.read_csv(raw_data_path + 'sample_submission.csv')

    def result_fill(x):
        return test_result_dict[x[0]]

    df_submission['has_cactus'] = df_submission.apply(result_fill, axis=1)
    df_submission.sort_values(by=['id']).to_csv(raw_data_path + 'submissions/submission.csv')

    return


if __name__ == '__main__':
    # look at versions to avoid confusion with deprecation/compatibility
    list_packages_versions()

    # do some proprocessing to make it compatible with the flow_from_directory() method
    raw_data_path = 'data/'
    data_raw2ImageDataGenerator(raw_data_path)

    # model the training data with simple 2DConvNet from scratch
    model_cactus = build_model(raw_data_path)

    # get model results
    probabilities = test_model(raw_data_path, model_cactus)

    # write model results in the format required by the Kaggle competition.
    write_submission(raw_data_path, probabilities)
