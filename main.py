import os
import tensorflow as tf
import numpy as np


# Set the seed for random operations.
# This let our experiments to be reproducible.
SEED = 1234
tf.random.set_seed(SEED)

# Get current working directory
cwd = os.getcwd()

# Set GPU memory growth
# Allows to only as much GPU memory as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# ImageDataGenerator
# ------------------
from keras.preprocessing.image import ImageDataGenerator


apply_data_augmentation = False

# Create training ImageDataGenerator object
if apply_data_augmentation:
    train_data_gen = ImageDataGenerator(rotation_range=10,
                                        width_shift_range=10,
                                        height_shift_range=10,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        cval=0,
                                        rescale=1. / 255)
else:
    train_data_gen = ImageDataGenerator(rescale=1. / 255)

# Create validation and test ImageDataGenerator objects
valid_data_gen = ImageDataGenerator(rescale=1. / 255)
test_data_gen = ImageDataGenerator(rescale=1. / 255)

# Create generators to read images from dataset directory
# -------------------------------------------------------
dataset_dir = os.path.join(cwd, 'data/New_Classification_Dataset')

# Batch size
bs = 8

# img shape
img_h = 256
img_w = 256

num_classes=20

decide_class_indices = False

if decide_class_indices:
    classes = ['airplanes',         # 0
               'bear',              # 1
               'calculator',        # 2
               'computer-monitor',  # 3
               'fireworks',         # 4
               'galaxy',            # 5
               'grand-piano',       # 6
               'kangaroo',          # 7
               'laptop',            # 8
               'lightbulb',         # 9
               'lightning',         # 10
               'mountain-bike',     # 11
               'owl',               # 12
               'school-bus',        # 13
               'sheet-music',       # 14
               'skyscraper',        # 15
               'sword',             # 16
               't-shirt',           # 17
               'waterfall',         # 18
               'wine-bottle']       # 19
else:
    classes=None

# Training
training_dir = os.path.join(dataset_dir, 'training')
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               classes=classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED)  # targets are directly converted into one-hot vectors

# Validation
validation_dir = os.path.join(dataset_dir, 'validation')
valid_gen = valid_data_gen.flow_from_directory(validation_dir,
                                               batch_size=bs,
                                               classes=classes,
                                               class_mode='categorical',
                                               shuffle=False,
                                               seed=SEED)

# Test
test_dir = os.path.join(dataset_dir, 'test')
test_gen = test_data_gen.flow_from_directory(test_dir,
                                             batch_size=bs,
                                             classes=classes,
                                             class_mode='categorical',
                                             shuffle=False,
                                             seed=SEED)

# Create Dataset objects
# ----------------------

# Training
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

# Shuffle (Already done in generator..)
# train_dataset = train_dataset.shuffle(buffer_size=len(train_gen))

# Normalize images (Already done in generator..)
# def normalize_img(x_, y_):
#     return tf.cast(x_, tf.float32) / 255., y_

# train_dataset = train_dataset.map(normalize_img)

# 1-hot encoding <- for categorical cross entropy (Already done in generator..)
# def to_categorical(x_, y_):
#     return x_, tf.one_hot(y_, depth=10)

# train_dataset = train_dataset.map(to_categorical)

# Divide in batches (Already done in generator..)
# train_dataset = train_dataset.batch(bs)

# Repeat
# Without calling the repeat function the dataset
# will be empty after consuming all the images
train_dataset = train_dataset.repeat()

# Validation
# ----------
valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

# Repeat
valid_dataset = valid_dataset.repeat()

# Test
# ----
test_dataset = tf.data.Dataset.from_generator(lambda: test_gen,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

# Repeat
test_dataset = valid_dataset.repeat()

print(train_gen.class_indices)

# Keras Model subclassing
# -----------------------

# Please note that there are many ways to implement a CNN
# Here subclassing is used just for teaching purposes, but you
# can simply write every single layer as usual using Model or Sequential..

# Create convolutional block
class ConvBlock(tf.keras.Model):
    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(filters=num_filters,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding='same')
        self.activation = tf.keras.layers.ReLU()  # we can specify the activation function directly in Conv2D
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.activation(x)
        x = self.pooling(x)
        return x


# Create Model
# ------------

depth = 5
start_f = 8
num_classes = 20


class CNNClassifier(tf.keras.Model):
    def __init__(self, depth, start_f, num_classes):
        super(CNNClassifier, self).__init__()

        self.feature_extractor = tf.keras.Sequential()

        for i in range(depth):
            self.feature_extractor.add(ConvBlock(num_filters=start_f))
            start_f *= 2

        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.Dense(units=512, activation='relu'))
        self.classifier.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    def call(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# Create Model instance
model = CNNClassifier(depth=depth,
                      start_f=start_f,
                      num_classes=num_classes)
# Build Model (Required)
model.build(input_shape=(None, img_h, img_w, 3))

# Visualize created model as a table
model.feature_extractor.summary()

# Visualize initialized weights
model.weights

# Optimization params
# -------------------

# Loss
loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

import os
from datetime import datetime

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

cwd = os.getcwd()

exps_dir = os.path.join(cwd, 'classification_experiments')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

model_name = 'CNN'

exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

callbacks = []

# Model checkpoint
# ----------------
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'),
                                                   save_weights_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)

# Visualize Learning on Tensorboard
# ---------------------------------
tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)

# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks.append(tb_callback)

# Early Stopping
# --------------
early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    callbacks.append(es_callback)

model.fit(x=train_dataset,
          epochs=100,  # set repeat in training dataset
          steps_per_epoch=len(train_gen),
          validation_data=valid_dataset,
          validation_steps=len(valid_gen),
          callbacks=callbacks)

# How to visualize Tensorboard

# 1. tensorboard --logdir EXPERIMENTS_DIR --port PORT     <- from terminal
# 2. localhost:PORT   <- in your browser

# model.load_weights('/path/to/checkpoint')  # use this if you want to restore saved model

# eval_out = model.evaluate(x=test_dataset,steps=len(test_gen),verbose=0)

# print(eval_out)

from PIL import Image

shoe_img = Image.open("/Users/Stefano/anndl/data/New_Classification_Dataset/test/IMG_32.jpg").convert('L')

shoe_arr = np.expand_dims(np.array(shoe_img), 0)

out_softmax = model.predict(x=shoe_arr / 255.)

out_softmax = tf.keras.activations.softmax(tf.convert_to_tensor(out_softmax))

predicted_class = tf.argmax(out_softmax, 1)

print(predicted_class)
