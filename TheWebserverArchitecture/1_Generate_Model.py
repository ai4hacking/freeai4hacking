#AI4HACKING.com
#Thanks to Google, extracted from: https://www.tensorflow.org/datasets/keras_example
#Enriched by Raúl Moreno Izquierdo
#Dataset: The MNIST (Modified National Institute of Standards and Technology) data consists of 60,000 training images and 10,000 test images. Each image is a crude 28 x 28 (784 pixels) handwritten digit from "0" to "9." Each pixel value is a grayscale integer between 0 and 255.
#Model metrics: Accuracy: 0.9877, Sparse_categorical_accuracy: 0.9861, Loss: 0.0451

from tokenize import Number
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

#Load dataset example 
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

#FOR TRAIN SUBSET:
#TFDS provides images of type tf.uint8 , while the model expects tf.float32 . Therefore, you need to normalize the images.
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

#As you adjust the dataset in memory, cache it before shuffling for better performance.
#Note: Random transformations must be applied after caching.
ds_train = ds_train.cache()

#For true randomness, set the shuffle buffer to the full size of the dataset.
#Note: For large data sets that don't fit in memory, use buffer_size=1000 if your system allows it.
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)

#Batch elements of the dataset after shuffling to get unique batches at each epoch.
ds_train = ds_train.batch(128)

#It is good practice to terminate the pipeline by performance prefetch.
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

#FOR TEST SUBSET:
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

#Plug the TFDS input pipeline into a simple Keras model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

#Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

#Train the model
history = model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

# Save the model
model.save('my_model_MNIST_AI4HACKING.h5')
model.save_weights('my_weights__MNIST_AI4HACKING')

# Generate Cross-entropy loss figure
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Cross-entropy loss")
plt.legend()
plt.savefig('Cross-entropy-loss.png')

#The model evaluate function predicts the output for the given input and then computes the metrics function specified in the model. 
loss,accuracy = model.evaluate(ds_train)
print('Accuracy is: ', accuracy)
print('Loss is: ', loss)

#Perform preditions in order to get the Confusion Matrix
predictions = model.predict(ds_test)

#Select the prediction for each case
testPredict = np.argmax(predictions, axis=1)

# Then, take all the y values from the prefetch dataset (thus changing the shape), equivalent to select all different classes:
trueClasses = tf.concat([y for x, y in ds_test], axis=0)

#Generate the confusion matrix for evaluation of metrics
cf_matrix = confusion_matrix(trueClasses, testPredict)

print("------CONFUSION MATRIX------")
print(cf_matrix)







