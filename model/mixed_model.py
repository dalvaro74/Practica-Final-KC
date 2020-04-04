import numpy as np
import pandas as pd
import urllib.request
import logging
import pickle
from util.models import *

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

from keras.optimizers import Adam
from keras.layers import concatenate


# Initialize logging
logging.basicConfig(filename="logs/mixed_model.log",format='%(asctime)s %(message)s')
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)

my_logger.info('\n******************** COMIENZA MIXED MODEL ********************')

train = pd.read_csv('model/data/train_model.csv',sep=';', decimal='.')
test = pd.read_csv('model/data/test_model.csv',sep=';', decimal='.')

#Generación de los tensores que representaran las distintas imagenes

import cv2
from time import time
array_images_train = train['image_path'].values
array_images_test = test['image_path'].values
N_train = len(array_images_train)
N_test = len(array_images_test)
# Esto esta mal considerando que TensorFlow es por defecto "Channel Last" y que por tanto Keras( que usa Tensorflow por defecto) tambien lo será
#data = np.empty((N, 3, 144, 216), dtype=np.uint8)
data_img_train= np.empty((N_train, 144, 216, 3), dtype=np.uint8) # Y asi no deberiamos necesitar transponer
data_img_test= np.empty((N_test, 144, 216, 3), dtype=np.uint8) # Y asi no deberiamos necesitar transponer
start_time = time()
# Debemos tener en cuenta que imread devuelve (alto, ancho, chanels)

# Obtenemos el array de imagenes de train 
for i, fpath in enumerate(array_images_train):
  if i%100 == 0:
    print('LLevamos {} Registros tratados de train'.format(i))
  img = cv2.imread(fpath, cv2.IMREAD_COLOR)
  #data[i, ...] = img.transpose(2, 0, 1)
  # Ya no es necesario trasponer
  data_img_train[i, ...] = img

# Obtenemos el array de imagenes de test 
for i, fpath in enumerate(array_images_test):
  if i%100 == 0:
    print('LLevamos {} Registros tratados de test'.format(i))
  img = cv2.imread(fpath, cv2.IMREAD_COLOR)
  #data[i, ...] = img.transpose(2, 0, 1)
  # Ya no es necesario trasponer
  data_img_test[i, ...] = img

elapsed_time = time() - start_time
print("Elapsed time: %0.10f seconds." % elapsed_time)

#Normalizamos los pixeles al rango [0,1]
data_img_train=data_img_train/255
data_img_test=data_img_test/255


# El siguiente paso es preparar los dataset
'''
El dataset contiene los siguientes campos:
Barrio
Room Type
Accommodates
Bathrooms
Cleaning Fee	
Extra People
Minimum Nights	
Availability 30	
Review Scores Rating
Cancellation Policy
image_path	

Para entrenar la red Neuronal nos quedamos con los siguientes:
Barrio, Room Type, Accommodates, Bathrooms, Cleaning Fee, Extra People
		
'''
y_train = train['Price']
X_train= train.drop(['Price','Minimum Nights', 'Availability 30', 'Review Scores Rating', 'Review Scores Rating', 'Cancellation Policy', 'image_path' ], axis=1)
y_test = test['Price']
X_test= test.drop(['Price','Minimum Nights', 'Availability 30', 'Review Scores Rating', 'Review Scores Rating', 'Cancellation Policy', 'image_path' ], axis=1)

print(f'Dimensiones del dataset de training: {X_train.shape}')
print(f'Dimensiones del dataset de test: {X_test.shape}')
# MIXED CNN

# Creamos las redes nn y cnn
nn = create_nn(X_train.shape[1], regress=False)
cnn = create_cnn(216, 144, 3, regress=False)

# Concatenamos la salida de cnn y nn que se convertira en la entrada de nuestro conjunto final de capas de prediccion
combinedInput = concatenate([nn.output, cnn.output])

# La capa final "Fully Connected" estara formada por dos capas densas, la ultima sera la que llevara a cabo la prediccion.
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

# Nuestro modelo final aceptara datos categoricos/numericos en la red tradicional (nn)
# e imagenes en la red convolucional (cnn), generando un valor escalar que 
#representara la prediccion del valor de la casa 
model = Model(inputs=[nn.input, cnn.input], outputs=x)

# Compilamos el modelo usando "mean_absolute_percentage_error" como funcion de perdida
opt = Adam(lr=0.001, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# Entrenamos el modelo
#Usamos los datos categoricos/numericos sin normalizar porque como hemos podido comprobar en nn_regresion_02 dan mejor resultado que los normalizados
n_epochs = 2
print("[INFO] training model...")
history_callback = model.fit(
	[X_train, data_img_train], y_train,
	validation_data=([X_test, data_img_test], y_test),
	epochs=n_epochs, batch_size=64)

# Guardamos nuestro modelo para poder usarlo mas adelante
pickle.dump(model, open('web/pkls/mixed_model.pkl', 'wb'))

#Para cargar el modelo basta con
#MODEL = pickle.load(open('model.pkl', 'rb'))

# veamos nuestra curva de pérdidas
'''
plt.plot(np.arange(0, n_epochs), history_callback.history["loss"])
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
'''

# Bondad del modelo
# Obtenemos la prediccion de nuestro modelo para los datos de Test
preds = model.predict([X_test, data_img_test])
 

# Calulamos la diferencia entre los precios predichos y los reales
# A continuacion calcularemos la diferencia porcentual y la diferecia absouta en porcentaje
diff = preds.flatten() - y_test
percentDiff = (diff / y_test) * 100
absPercentDiff = np.abs(percentDiff)
 

#Calculamos la media y la desviacion estandar
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
 
# Mostramos las estadisticas de nuestro modelo.
print("Precio medio de las pisos: {:.2f}€, Desviacion standar: {:.2f}€".format(
    train["Price"].mean(), train["Price"].std()))
print("Modelo --> mean Error: {:.2f}%, std Error: {:.2f}%".format(mean, std))

