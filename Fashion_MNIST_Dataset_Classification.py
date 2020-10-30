import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt

# defining callback class
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.85):
      print("cancelling training")
      self.model.stop_training = True


def plot_history(net_history):
    history = net_history.history
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['accuracy']
    val_accuracies = history['val_accuracy']
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])
    
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['accuracy', 'val_accuracy'])

# loading dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("train_images dimentions: ", train_images.ndim)
print("train_images shape: ", train_images.shape)
print("train_images type: ", train_images.dtype)

print("test_images dimentions: ", test_images.ndim)
print("test_images shape: ", test_images.shape)
print("test_images type: ", test_images.dtype)

X_train = train_images.reshape(60000, 784)
X_test = test_images.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)

# creating model
myModel = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),  
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

myModel.summary()
myModel.compile(optimizer=SGD(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

# training model
callbacks = myCallback()
network_history = myModel.fit(X_train, Y_train, batch_size=128, epochs=1000, callbacks=[callbacks], validation_split=0.2)
plot_history(network_history)