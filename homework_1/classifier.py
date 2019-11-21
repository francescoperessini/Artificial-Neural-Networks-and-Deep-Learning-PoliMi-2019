import tensorflow as tf
import numpy as np
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from homework_1.csv_builder import create_csv

SEED = 1234
tf.random.set_seed(SEED)

# Get current working directory
cwd = os.getcwd()

# ImageDataGenerator
# ------------------
apply_data_augmentation = True

# Create training ImageDataGenerator object
if apply_data_augmentation:
    train_data_gen = ImageDataGenerator(rotation_range=20,
                                        width_shift_range=10,
                                        height_shift_range=10,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=False,
                                        fill_mode='nearest',
                                        cval=0,
                                        rescale=1. / 255,
                                        brightness_range=[0.5, 1.5],
                                        shear_range=0.7)
else:
    train_data_gen = ImageDataGenerator(rescale=1. / 255)

# Create validation and test ImageDataGenerator objects
valid_data_gen = ImageDataGenerator(rescale=1. / 255)
test_data_gen = ImageDataGenerator(rescale=1. / 255)

# Create generators to read images from dataset directory
# -------------------------------------------------------
# dataset_relative_dir = "../input/New_Classification_Dataset"
# dataset_dir = dataset_relative_dir

dataset_test_dir = os.path.join(cwd, "../homework_1/data/New_Classification_Dataset")

# Batch size
bs = 8

# img shape
img_h = 300
img_w = 300

num_classes = 20

decide_class_indices = True
if decide_class_indices:
    classes = ['owl',               # 0
               'galaxy',            # 1
               'lightning',         # 2
               'wine-bottle',       # 3
               't-shirt',           # 4
               'waterfall',         # 5
               'sword',             # 6
               'school-bus',        # 7
               'calculator',        # 8
               'sheet-music',       # 9
               'airplanes',         # 10
               'lightbulb',         # 11
               'skyscraper',        # 12
               'mountain-bike',     # 13
               'fireworks',         # 14
               'computer-monitor',  # 15
               'bear',              # 16
               'grand-piano',       # 17
               'kangaroo',          # 18
               'laptop']            # 19
else:
    classes = None

# Training
training_dir = os.path.join(dataset_test_dir, 'training')
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               classes=classes,
                                               class_mode='categorical',
                                               target_size=(img_h, img_w),
                                               shuffle=True,
                                               seed=SEED)  # targets are directly converted into one-hot vectors

# Validation
validation_dir = os.path.join(dataset_test_dir, 'validation')
valid_gen = valid_data_gen.flow_from_directory(validation_dir,
                                               batch_size=bs,
                                               classes=classes,
                                               class_mode='categorical',
                                               target_size=(img_h, img_w),
                                               shuffle=False,
                                               seed=SEED)

# Create Dataset objects
# ----------------------
# Training
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
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
        self.classifier.add(tf.keras.layers.Dropout(0.5))
        self.classifier.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    def call(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


weight_file = '../data/trained_model.ckpt'

# Create Model instance
model = CNNClassifier(depth=depth,
                      start_f=start_f,
                      num_classes=num_classes)

# Build Model (Required)
model.build(input_shape=(None, img_h, img_w, 3))
loss = tf.keras.losses.CategoricalCrossentropy()
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# Compile Model
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

callbacks = []

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weight_file,
                                                   save_weights_only=True,
                                                   verbose=1)  # False to save the model directly
callbacks.append(ckpt_callback)

early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    callbacks.append(es_callback)

model.fit(x=train_dataset,
          epochs=1,  # set repeat in training dataset
          steps_per_epoch=len(train_gen),
          validation_data=valid_dataset,
          validation_steps=len(valid_gen),
          callbacks=callbacks)


dataset_test_dir = "../homework_1/data/New_Classification_Dataset/test"
sub_files = os.listdir(dataset_test_dir)

results = {}

for file in sub_files:
    test_image = os.path.join(dataset_test_dir, file)
    img = Image.open(test_image).convert('RGB')
    img = img.resize((img_h, img_w))
    arr = np.expand_dims(np.array(img), 0)

    out_softmax = model.predict(x=arr / 255.)

    predicted_class = tf.argmax(out_softmax, 1)

    class_name = predicted_class.numpy()[0]
    results[file] = class_name

create_csv(results)
