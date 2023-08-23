import zipfile
import os
import time
import shutil
from google.colab import drive
from PIL import Image
import tensorflow as tf
#import tensorflow_datasets as tfds
from keras.utils.vis_utils import plot_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
from tqdm import tqdm

#смонтировать google disk в colab
drive.mount('/content/drive')

#изменить рабочую папку на ту, где хранятся файлы модели
print(os.getcwd())
os.chdir('/content/drive/MyDrive/Classification/mnist')
print(os.getcwd())

#распаковка архивов с датасетами
shutil.unpack_archive('/content/drive/MyDrive/Classification/mnist/IMG/train/0/0 train.zip', '/content/drive/MyDrive/Classification/mnist/IMG/train/0')
shutil.unpack_archive('/content/drive/MyDrive/Classification/mnist/IMG/train/1/1 train.zip', '/content/drive/MyDrive/Classification/mnist/IMG/train/1')
shutil.unpack_archive('/content/drive/MyDrive/Classification/mnist/IMG/test/0/0 test.zip', '/content/drive/MyDrive/Classification/mnist/IMG/test/0')
shutil.unpack_archive('/content/drive/MyDrive/Classification/mnist/IMG/test/1/1 test.zip', '/content/drive/MyDrive/Classification/mnist/IMG/test/1')

#количество изображений, при котором изменяются значения весов и обучение
batch_size = 128
image_size = (256, 256)
num_classes = 2

#указывает на подкаталоги рабочей папки, в которй хранятся изображения
image_tr = os.path.join("IMG","train")
image_val = os.path.join("IMG","test")

#генерирует датасет tensorflow из указанной ранее папки с изображениями
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_tr,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_val,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size
)

#количество итераций обновления весов в одной эпохе
num_iterations = int(len(train_ds.file_paths)/batch_size) + 1

#возвращает объект итератора(содержит изображения), выдает их по запросу
train_ds = train_ds.map(lambda image, label: (tf.image.resize(image, image_size), label))
val_ds = val_ds.map(lambda image, label: (tf.image.resize(image, image_size), label))

print(len(train_ds))

#увеличение количества данных для обучения, за счет модификации уже существующих изображений(повороты, искривления)
img_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomContrast(factor=0.1),

    ],
    name="img_augmentation",
)

#использование "горячего кодирования" данных для категорий изображений
def input_preprocess(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

#возвращает итерируемый объект с изображениями, обработанными горячим кодированием
#создание
train_ds = train_ds.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)

#использует метод .prefetch, чтробы снизить время простоя gpu во время подготовки cpu данных к обучению на gpu
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(input_preprocess)

#создание тензора, основываясь на размере изображений
inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
x = img_augmentation(inputs)

#загрузка уже заранее обученной модели EfficientNetB0 на ImageNet
model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=x, weights="noisy.student.notop-b0.h5", classes=num_classes)
#model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=x, classes=num_classes)

# Freeze the pretrained weights
model.trainable = False

# Rebuild top
x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = tf.keras.layers.BatchNormalization()(x)

#случайно обнуляет заданный процент значений весов нейронов
top_dropout_rate = 0.6
x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

model = tf.keras.Model(inputs, outputs, name="EfficientNet")

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#print(model.summary())

#число, на которое изменяется значение весов после 1 шага обучения
learning_rate = 1e-4

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    #learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


unfreeze_model(model)

epochs = 10
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=2, batch_size=batch_size)


model.save(f'E:{epochs}_LR:{learning_rate}_B:{batch_size}.h5')

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.xticks(np.arange(epochs))
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_hist(hist)