import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
  read data from .csv file 
'''
samples = []
labels = []

with open('data.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
for i in data:
    samples.append(float(i[0]))
    labels.append(int(i[1]))

trainSamples = np.array(samples[:len(samples)//2])
trainLabels = np.array(labels[:len(samples)//2])
testSamples = np.array(samples[len(samples)//2:])
testLabels = np.array(labels[len(labels)//2:])

'''
  make model
'''
model = tf.keras.Sequential([
    Dense(units=16, input_shape=(1,), activation='sigmoid'),
    Dense(units=32, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=2, activation='softmax')
])
model.summary()

'''
  train model
'''
model.compile(optimizer=Adam(learning_rate=0.0007), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=trainSamples, y=trainLabels, validation_split=0.1, batch_size=15, epochs=100, shuffle=True, verbose=2)

'''
  predictions
'''
predictions = model.predict(x=testSamples, batch_size=10, verbose=0)

roundedPredictions = np.argmax(predictions, axis=-1)

cm = confusion_matrix(y_true = testLabels, y_pred = roundedPredictions)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix,', cmap=mpl.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:np.newaxis]
        print('Normalized confusiom matrix')
    else:
        print('Confusion matrix without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], 
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Prawdziwe')
    plt.xlabel('Przewidywane')
    plt.show()

cm_plot_labels = ['nie para', 'para']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Macierz pomyłek")