import numpy as np
from keras.applications import VGG16
from keras.applications.densenet import layers
from keras.dtensor import optimizers
from keras.layers import GlobalMaxPooling2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pandas as pd
from keras.models import Model

# Hyperparameters
epochs = 30
batch_size = 64

def input_name():
    filenames = os.listdir("AutismDataset/train")
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'Autistic':
            categories.append(str(1))
        else:
            categories.append(str(0))

    train_df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    test_filenames = os.listdir("AutismDataset/test")
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'Autistic':
            categories.append(str(1))
        else:
            categories.append(str(0))

    test_df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    prepare_model(train_df, test_df)



# uncomment this code To See image
# sample = random.choice(filenames)
# image = load_img("AutismDataset/train/"+sample)
# plt.imshow(image)

def model_summary():
    image_size = 224
    input_shape = (image_size, image_size, 3)

    pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

    last_layer = pre_trained_model.get_layer('block5_pool')
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension
    x = GlobalMaxPooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.5
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(2, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])

    model.summary()


def prepare_model(train_df, image_size,batch_size):
    model_summary()
    # Prepare Test and Train Data
    train_df, validate_df = train_test_split(train_df, test_size=0.1)
    train_df = train_df.reset_index()
    validate_df = validate_df.reset_index()

    # validate_df = validate_df.sample(n=100).reset_index() # use for fast testing code purpose
    # train_df = train_df.sample(n=1800).reset_index() # use for fast testing code purpose

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        "AutismDataset/train/",
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=(image_size, image_size),
        batch_size=batch_size
    )
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        "AutismDataset/train/",
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=(image_size, image_size),
        batch_size=batch_size
    )
    # Fit Model
    history = Model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate // batch_size,
        steps_per_epoch=total_train // batch_size)

    loss, accuracy = Model.evaluate_generator(validation_generator, total_validate // batch_size, workers=12)
    print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))
    show_grafik(history)

def save_model():
    prepare_model()
    Model.save('vgg19.h5')

def show_grafik(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))

    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()

