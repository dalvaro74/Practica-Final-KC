import numpy as np
import pandas as pd
import logging
import pickle
from util.models import *

# import the necessary packages
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical




# Initialize logging
logging.basicConfig(filename="logs/nn_model.log",format='%(asctime)s %(message)s')
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)

my_logger.info('\n******************** COMIENZA NN MODEL ********************')

train = pd.read_csv('model/data/train_model.csv',sep=';', decimal='.')
test = pd.read_csv('model/data/test_model.csv',sep=';', decimal='.')

# El siguiente paso es preparar los dataset
'''
El dataset contiene los siguientes campos:
Barrio
Distrito
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

Para entrenar la red Neuronal nos quedamos con los siguientes (para no pedirle demasiados campos al usuario):
Barrio, Room Type, Accommodates, Bathrooms
		
'''
y_train = train['Price']
X_train= train.drop(['Price','Minimum Nights', 'Availability 30', 'Review Scores Rating', 'Review Scores Rating', 'Cancellation Policy', 'image_path', 'Distrito', 'Cleaning Fee', 'Extra People' ], axis=1)
y_test = test['Price']
X_test= test.drop(['Price','Minimum Nights', 'Availability 30', 'Review Scores Rating', 'Review Scores Rating', 'Cancellation Policy', 'image_path', 'Distrito', 'Cleaning Fee', 'Extra People' ], axis=1)

print(f'Dimensiones del dataset de training: {X_train.shape}')
print(f'Dimensiones del dataset de test: {X_test.shape}')
y_test.isna().sum()

# CREAMOS EL MODELO

# Creamos lar neuronal
nn_model = create_nn(X_train.shape[1], regress=True)

#Ponemos el numero de epocas como una variable
n_epochs = 50

# Compilamos el modelo usando "mean_absolute_percentage_error" como funcion de perdida
opt = Adam(lr=0.01, decay=1e-3)
#nn_model.compile(loss="mean_absolute_percentage_error", optimizer=opt, metrics=['mae'])
nn_model.compile(loss="mean_absolute_error", optimizer=opt)

# Entrenamos el modelo
#Usamos los datos categoricos/numericos sin normalizar porque como hemos podido comprobar en nn_regresion_02 dan mejor resultado que los normalizados

print("[INFO] training model...")
history_callback  = nn_model.fit(X_train, y_train,
batch_size=32,
#validation_split=0.1,
shuffle=True,
epochs=n_epochs,
validation_data=(X_test, y_test))


# Guardamos nuestro modelo para poder usarlo mas adelante
pickle.dump(nn_model, open('web/pkls/nn_model.pkl', 'wb'))

#Para cargar el modelo basta con
#MODEL = pickle.load(open('nn_model.pkl', 'rb'))

# veamos nuestra curva de pérdidas
'''
plt.plot(np.arange(0, n_epochs), history_callback.history["loss"])
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
'''

# Bondad del modelo
# Obtenemos la prediccion de nuestro modelo para los datos de Test
preds = nn_model.predict(X_test)


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

