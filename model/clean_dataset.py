'''
En este script vamos a partir del dataset (airbnb-images-clean.csv)

Este dataset ha sido creado en el script "descarga_imagenes" con aquellos registros que contengan una imagen valida para poder ser usado en el modelo convolucional que crearemos para el caso (notebook cnn_regresion)

En este caso vamos a usar el codigo del script models.py extraido de:

https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

Dicho código ha sido modificado para adaptarse a nuestras necesidades.
'''

#Imports
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


import warnings
warnings.filterwarnings('ignore')
import logging
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from util.utilidades import *

# Initialize logging
#fn = os.path.join(os.path.dirname(__file__), 'clean_dataset.log')
logging.basicConfig(filename='logs/clean_dataset.log', format='%(asctime)s %(message)s')
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)

my_logger.info('\n******************** COMIENZA PROCESO DE LIMPIEZA DEL DATASET ********************')

# Cargamos los datos del fichero de airbnb reducido. Este fichero contiene 11925 observaciones y 90 variables
#fn = os.path.join(os.path.dirname(__file__), 'data/airbnb-images-clean.csv')
#full_airbnb_images = pd.read_csv(fn, sep=';', decimal='.')
full_airbnb_images = pd.read_csv('model/data/airbnb-images-clean.csv',sep=';', decimal='.')

my_logger.info(f'Dimensiones del dataset al inicio: {full_airbnb_images.shape}')

#################### LIMPIEZA GENERAL QUE AFECTA AL CONJUNTO DEL DATASET ####################

# Eliminamos los Outliers
full_airbnb_images = full_airbnb_images[(full_airbnb_images['Price']>10) & (full_airbnb_images['Price']<200)]

# Para hacer el analisis de los pisos de Madrid vamos a eliminar todos los que no tengan "Neighbourhood Group Cleansed" 
# Tambien los que correspondan a "Neighbourhood Group Cleansed" que se repitan menor de 30 veces
# ya que hemos comprobado que se corresponden con barrios fuera de Madrid

#Nans
full_airbnb_images = full_airbnb_images[full_airbnb_images['Neighbourhood Group Cleansed'].notna()]
#Eliminados 653 registros

#Frecuencia menor de 30
neig_group_tmp = full_airbnb_images['Neighbourhood Group Cleansed']
freqs = neig_group_tmp.value_counts()
min_freq = 30
freqs2remove = freqs[freqs<=min_freq].index
full_airbnb_images = full_airbnb_images[~full_airbnb_images['Neighbourhood Group Cleansed'].isin(freqs2remove)]
#Eliminados 138 registros

# Quedan 10674
#full_airbnb_images.shape

# Por ultimo me cargo barrios que se que no son de Madrid sino de barceloana
barrios_barcelona = ['Eixample', 'Sants-Montjuïc', 'Gràcia', 'Ciutat Vella']
full_airbnb_images = full_airbnb_images[~full_airbnb_images['Neighbourhood Group Cleansed'].isin(barrios_barcelona)]
#Eliminados 196 registros
# Quedan 10478

'''
Antes de llevar a cabo el split entre Trainintg y Test eliminamos las columnas que tenemos claro que no van a ser utiles para nuestro objetivo:
1. **Las que contienen URLs**: Listing Url: drop_url  
2. **Los Ids y lo relativo al Scrape realizado**: drop_id_scrape
3. **Nombres y comentarios**:drop_comments
4. **Direcciones**: A la vista de la informacion que contienen las variables de direccion podemos dropear varias de ellas por diversos motivos
 (sin que tengamos que dividir previamente en Train Test). 
 Demasiado genericas: City, State, Market, Smart Location, Country Code, Country, Jurisdiction Names. 
 Demasiado concretas: Street, Latitude, Longitude y Geolocation. 
 Demasiado ruido o demasiados registros nulos: Neighbourhood, Host Location, Host Neighbourhood. 
 Por ultimo, Zipcode es una variable que para representar la direccion no me parece la mas adecuada debido a que aunque es un numero, 
 deberia ser tratado como una variable categorica. 
 Ademas contiene bastante ruido, una cantidad no despreciable de nulos y tambien es demasiado concreta (506 valores unicos)     
 Por tanto y para la evaluacion del modelo deberemos barajar cual de las dos opciones que quedan es la mejor para representar
 la "zona" en la que se encuentra el piso ( Neighbourhood Cleansed o Neighbourhood Group Cleansed, las cuales obviamente van a estar fuertemente correladas), 
 pero esto debera hacerse una vez separado el dataset, para que los datos de Test no influyan en la decision. 
 (En cualquier caso sera necesario hacer un trabajo de limpieza y categorizacion con la variable elegida):drop_address
5. **Informascion relativa al Hospedador**: drop_host
6. **Nuevo campo incluido en la descarga de imagenes**: image_path
'''
drop_url = np.array(['Listing Url', 'Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url', 'Host URL',
                     'Host Thumbnail Url','Host Picture Url'])
full_airbnb_images.drop(drop_url, axis=1, inplace=True)

drop_id_scrape = np.array(['ID', 'Scrape ID', 'Last Scraped', 'Host ID', 'Calendar last Scraped'])
full_airbnb_images.drop(drop_id_scrape, axis=1, inplace=True)


drop_comments = np.array(['Name', 'Summary', 'Space', 'Description', 'Neighborhood Overview', 'Notes',
                     'Transit','Access', 'Interaction', 'House Rules', 'Host Name', 'Experiences Offered',
                         'Host About', 'Amenities', 'Features'])

full_airbnb_images.drop(drop_comments, axis=1, inplace=True)

drop_address = np.array(['Host Location', 'Host Neighbourhood', 'Neighbourhood', 'Street', 'Zipcode', 
    'City', 'State', 'Market', 'Smart Location','Country Code', 'Country', 'Latitude', 
                         'Longitude', 'Jurisdiction Names', 'Geolocation'])

full_airbnb_images.drop(drop_address, axis=1, inplace=True)

drop_host = np.array(['Host Since', 'Host Response Time', 'Host Response Rate', 'Host Acceptance Rate', 
    'Host Listings Count', 'Host Total Listings Count','Host Verifications', 'Calculated host listings count'])

full_airbnb_images.drop(drop_host, axis=1, inplace=True)


'''
Por ultimo eliminamos varios campos sueltos por los siguientes motivos:

Square Feet: Contiene 96% observaciones null
Weekly Price: Contiene 76% observaciones null
Monthly Price: Contiene 76% obsevaciones null
Has Availability: Contiene 99% observaciones null
First Review: No creemos que aporte informacion util para el calculo del precio
Last Review: No creemos que aporte informacion util para el calculo del precio
Calendar Updated: No creemos que aporte informacion util para el calculo del precio
License: Contiene 98% observaciones null
Bed Type: Casi el 99% de las camas son del mismo tipo (Real Bed)
'''

drop_varios = np.array(['Square Feet', 'Weekly Price', 'Monthly Price', 'Has Availability', 'First Review', 'Last Review',
                     'Calendar Updated','License', 'Bed Type'])

full_airbnb_images.drop(drop_varios, axis=1, inplace=True)

# Tras la limpeza inicial nos hemos quedado con 30 caracteristicas (y el target).

my_logger.info(f'Dimensiones del dataset despues de la limpieza general: {full_airbnb_images.shape}')

# Separamos train y Test y guardamos una primera version de ambos como csv
# Tengamos en cuenta que todas las transformaciones que se hagan sobre train se deberan hacer de igual manera sobre test

from sklearn.model_selection import train_test_split
train, test = train_test_split(full_airbnb_images, test_size=0.2, shuffle=True, random_state=0)
my_logger.info(f'Dimensiones del dataset de training: {train.shape}')
my_logger.info(f'Dimensiones del dataset de test: {test.shape}')
# Guardamos
#fn = os.path.join(os.path.dirname(__file__), 'data/train.csv')
#train.to_csv(fn, sep=';', decimal='.', index=False)
train.to_csv('model/data/train.csv', sep=';', decimal='.', index=False)
#fn = os.path.join(os.path.dirname(__file__), 'data/test.csv')
#test.to_csv(fn, sep=';', decimal='.', index=False)
test.to_csv('model/data/test.csv', sep=';', decimal='.', index=False)


###########
## TRAIN ##
###########

#Tratamiento de las variables categoricas

#Las varibales "Neighbourhood Group Cleansed" y "Neighbourhood Cleansed" ya han sido tratadas al principio (dejando solo pisos de Madrid) y no hay que tocarlas

# A continuación trataremos las otras tres variables categoricas que nos quedan (Property Type, Room Type y Cancellation Policy)
'''
Vamos a dejar las categorias de Property Type en "Apartment", "House", "Condominium", "Bed & Breakfast", "Loft", "Dorm", "Guesthouse",
"Chalet", "Townhouse", "Hostel" y "Villa" que representan mas del 95% del total y todas las demas las incluiremos en la categoría "Other".
Para ello usaremos la funcion change_cat_to_other del pakcage utilidades
'''
array_main_cat_property_type = ['Apartment', 'House', 'Condominium', 'Bed & Breakfast', 'Loft', 'Dorm', 'Guesthouse',
                               'Chalet', 'Townhouse', 'Hostel', 'Villa']
train['Property Type'] = change_cat_to_other(array_main_cat_property_type, train['Property Type'])


'''
De la misma manera vamos a dejar las categorias de Cancellation Policy en "strict", "flexible" y "moderate", que representan mas del 96% del total
y todas las demas las incluiremos en la categoría "Other". 
Usamos nuevamente la funcion change_cat_to_other
'''
array_main_cat_cancellation_policy = ['strict', 'flexible', 'moderate']
train['Cancellation Policy'] = change_cat_to_other(array_main_cat_cancellation_policy, train['Cancellation Policy'])

### MEAN ENCODING ###

'''
Una vez reducidas y limpiadas las categorias de las variables categoricas, las convertiremos en numericas mediante el mecanismo de mean encoding. 
Guardamos las transformacion hechas en Train para reproducirlas en Test con un replace o un map sin volver a aplicar el mean encoding 
para evitar que los datos de test infulyan en el modelo. Para aplicar le metodo Mean Encoding es conveniente que no haya NaNs en la variable Target (Price), 
por ello imputaremos esos valores con la media de los precios.
'''
y_train_mean = np.mean(train['Price'])
train['Price'] = train['Price'].fillna(y_train_mean)

#Property Type
mean_encode_property_type = train.groupby('Property Type')['Price'].mean()
train.loc[:,'Property Type'] = train['Property Type'].map(mean_encode_property_type)
# Guardamos el mean_encode para poder usarlo mas adelante
#fn = os.path.join(os.path.dirname(__file__), 'app/mean_encode_property_type.pkl')
#pickle.dump(mean_encode_property_type, open(fn, 'wb'))
pickle.dump(mean_encode_property_type, open('web/pkls/mean_encode_property_type.pkl', 'wb'))

#Cancellation Policy 
mean_encode_cancellation_policy = train.groupby('Cancellation Policy')['Price'].mean()
train.loc[:,'Cancellation Policy'] = train['Cancellation Policy'].map(mean_encode_cancellation_policy)
# Guardamos el mean_encode para poder usarlo mas adelante
#fn = os.path.join(os.path.dirname(__file__), 'app/mean_encode_cancellation_policy.pkl')
#pickle.dump(mean_encode_cancellation_policy, open(fn, 'wb'))
pickle.dump(mean_encode_cancellation_policy, open('web/pkls/mean_encode_cancellation_policy.pkl', 'wb'))

#Room Type
mean_encode_room_type = train.groupby('Room Type')['Price'].mean()
train.loc[:,'Room Type'] = train['Room Type'].map(mean_encode_room_type)
# Guardamos el mean_encode para poder usarlo mas adelante
#fn = os.path.join(os.path.dirname(__file__), 'app/mean_encode_room_type.pkl')
#pickle.dump(mean_encode_room_type, open(fn, 'wb'))
pickle.dump(mean_encode_room_type, open('web/pkls/mean_encode_room_type.pkl', 'wb'))

#Neighbourhood Cleansed
mean_encode_barrio = train.groupby('Neighbourhood Cleansed')['Price'].mean()
train.loc[:,'Neighbourhood Cleansed'] = train['Neighbourhood Cleansed'].map(mean_encode_barrio)
# Guardamos el mean_encode para poder usarlo mas adelante
#fn = os.path.join(os.path.dirname(__file__), 'app/mean_encode_barrio.pkl')
#pickle.dump(mean_encode_barrio, open(fn, 'wb'))
pickle.dump(mean_encode_barrio, open('web/pkls/mean_encode_barrio.pkl', 'wb'))

#Neighbourhood Group Cleansed
mean_encode_distrito = train.groupby('Neighbourhood Group Cleansed')['Price'].mean()
train.loc[:,'Neighbourhood Group Cleansed'] = train['Neighbourhood Group Cleansed'].map(mean_encode_distrito)
# Guardamos el mean_encode para poder usarlo mas adelante
#fn = os.path.join(os.path.dirname(__file__), 'app/mean_encode_distrito.pkl')
#pickle.dump(mean_encode_distrito, open(fn, 'wb'))
pickle.dump(mean_encode_distrito, open('web/pkls/mean_encode_distrito.pkl', 'wb'))

### CORRELACION DE VARIABLES
import seaborn as sns

# Compute the correlation matrix
corr = np.abs(train.drop(['Price'], axis=1).corr())
train
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# No pintaremos la grafica de correlaciones para que no interfiera, pero usaremos las conclusiones extraidas de ella para
# ver que variables podemos eliminar

# Set up the matplotlib figure
#f, ax = plt.subplots(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
#sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})
#plt.show()

'''
A la vista de la grafica de correlacion y siendo un poco generosos con las variables a eliminar para simplificar el modelo, 
podemos deducir lo siguiente: 
    1. Accommodates tiene una fuerte correlacion con Bedrooms, Beds y moderada con Guests Included 
    2. Availability 30 tiene fuerte correlacion con Availability 60 Availability 90 y moderada con Availability 365 
    3. Review Scores Rating tiene una fuerte correlacion con Review Scores Accuracy, Review Scores Cleanliness, Review Scores Checkin, 
    Review Scores Communication, Review Scores Value y moderada con Review Scores Location. 
    4. Number of Reviews tiene fuerte correlacion con Reviews per Month 5. Neighbourhood Cleansed y Neighbourhood Group Cleansed 
    muestran una fuerte correlacion, pero de momento dejamos las dos para analizar mediante el filtrado de caracteristcas 
    cual de las dos influye mas en la regresion que tenemos que plantear.
'''

#Vamos a eliminar todas estas variables del dataset de entrenamiento
drop_corr = ['Bedrooms', 'Beds', 'Guests Included', 'Availability 60', 'Availability 90', 'Availability 365',
             'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication',
             'Review Scores Value', 'Review Scores Location', 'Reviews per Month']
train.drop(drop_corr, axis=1, inplace=True)

my_logger.info(f'Dimensiones del dataset train tras la correlacion : {train.shape}')

'''
Tratamiento de NaNs (Imputacion de nulos)

Ningun modelo de Machine o Deep Learning funcionan adecuadamente sin un tratamiento previo de los NaNs de sus registros.
Por tanto debemos analizar que variables tienen valores NaN y llevemos a cabo el proceso de imputacion.
Contamos con la funion missing_values_table que nos permite visualizar con un unico comando los nulos que aparecen en los distintos campos del dataframe
'''

#my_logger.info(missing_values_table(train))

'''
Haciendo esto vemos que los unicos campos que contienen NaNs son:

Bathroom (26) Imputamos la media
Security Deposit (5391). Es mas de la mitad de los regitros por lo que lo eliminamos
Cleaning Fee (3785). Imputamnos un cero. asumismos que quien no tiene ese dato es porque no hay gastos de limpieza.
Review Scores Rating (1854). Imputamos la media. En su momento consideramos que el no tener este campo relleno
no era buena señal por que probamos a imputar el minimo, en lugar de la media, pero el modelo predecia peor, 
por lo que desestimamos nuestra hipotesis y volvimos a la media.
'''
mean_bathroom = np.mean(train['Bathrooms'])
mean_review = np.mean(train['Review Scores Rating'])
min_review = np.min(train['Review Scores Rating'])
train['Bathrooms'] = train['Bathrooms'].fillna(mean_bathroom)
train['Cleaning Fee'] = train['Cleaning Fee'].fillna(0)
train['Review Scores Rating'] = train['Review Scores Rating'].fillna(mean_review)

train.drop('Security Deposit', axis=1, inplace=True)

# Separemos el dataset train entre la varible dependiente (y_train) y el resto de variables independientes (X_train)
# Fundamentalmente para poder hacer el filtrado de caracteristicas para Regresion
y_train = train['Price']
X_train= train.drop(['Price','image_path'], axis=1)


'''
Filtrado para regresion
Con las 14 variables que me quedan en X_train aplico los metodos de fitrado f_regresion y mutual_info_regresion
'''

from sklearn.feature_selection import f_regression, mutual_info_regression
f_test, _ = f_regression(X_train, y_train)
f_test /= np.max(f_test)
mi = mutual_info_regression(X_train, y_train)
mi /= np.max(mi)

# Para mostrar esta informacion de forma grafica deberia hacer lo siguiente (comentado en este caso)
'''
featureNames = list(X_train.columns)

plt.figure(figsize=(30, 10))

plt.subplot(1,2,1)
plt.bar(range(X_train.shape[1]),f_test,  align="center")
plt.xticks(range(X_train.shape[1]),featureNames, rotation = 90)
plt.xlabel('features')
plt.ylabel('Ranking')
plt.title('$F-test$ score')


plt.subplot(1,2,2)
plt.bar(range(X_train.shape[1]),mi,  align="center")
plt.xticks(range(X_train.shape[1]),featureNames, rotation = 90)
plt.xlabel('features')
plt.ylabel('Ranking')
plt.title('Mutual information score')

plt.show()
'''

'''
A la vista de las graficas y de los valores de f_test y mi las variables que mas estan impactando en la variable objetivo son:

Neighbourhood Cleansed  mean_encode_distrito
Neighbourhood Group Cleansed
Room Type
Accommodates
Bathrooms
Cleaning Fee
Extra People
Minimum Nigths
Availability 30
cancelation Policy
Review Scores Rating
Como era de esperar el barrio, el tipo de habitacion y las personas que pueden ocupar la casa son los parametros que mas afectan al precio de la casa.

Por tanto seran estas las variables que usare para testear mis modelos, elimando el resto del dataset de training
'''
# Al modelo le pasaremos tanto Barrio como Distrito, y que elija el cual elegir
drop_filtrado = ['Property Type','Maximum Nights', 'Number of Reviews']
train.drop(drop_filtrado, axis=1, inplace=True)



# Por ultimo cambiare el nombre de la variable "Neighbourhood Cleansed" por "Barrio" 
# y "Neighbourhood Group Cleansed" por Distrito
train.rename(columns={'Neighbourhood Cleansed':'Barrio'}, inplace=True)
train.rename(columns={'Neighbourhood Group Cleansed':'Distrito'}, inplace=True)

my_logger.info(f'Dimensiones del dataset train tras el filtrado : {train.shape}')

# De momento la normalizacion nos la saltamos

###########
## TEST ##
###########
'''
Vamos a dejar preparado el dataset de Test con las mismas transformaciones que hemos llevado a cabo sobre el de Train. 
Abajo indicamos el listado de dichas transformaciones para no olvidarnos de ninguna:

Lo primero es el dropeo de las variables que no van a participar en el modelo (correlacion y filtrado): drop_corr y drop_filtrado.
Drop de "Security Deposit"
Tratamiento de las variables categoricas que influyen en el modelo (filtrado de categorias y Encoder): mean_encode_room_type, mean_encode_barrio y mean_encode_cancellation_policy
Imputacion de NaNs: Price, Bathroom y Cleaning Fee

Tengamos en cuenta que si cambiamos algo en el tratamiento del train deberemos incluirlo aqui
'''

#Dropeamos
test.drop(drop_corr, axis=1, inplace=True)
test.drop(drop_filtrado, axis=1, inplace=True)

#Eliminamos tambien la variable Security Deposit 
test.drop('Security Deposit', axis=1, inplace=True)

#imputamos los valores NaNs del target con la media del train (y_train_mean)
media_target_test = np.mean(test['Price'])
test['Price'] = test['Price'].fillna(y_train_mean)

#Imputamos los NaNs de la misma manera que se hace en Train
test['Bathrooms'] = test['Bathrooms'].fillna(mean_bathroom)
test['Cleaning Fee'] = test['Cleaning Fee'].fillna(0)
test['Review Scores Rating'] = test['Review Scores Rating'].fillna(mean_review)


#Aplicamos el Mean Encoder a "Room Type", "Neighbourhood Cleansed" "Neighbourhood Group Cleansed"
# y "Cancellation Policy"  el obtenido en train, si aparece alguna categoria nueva en test
# lo trataremos a posteriori
test.loc[:,'Room Type'] = test['Room Type'].map(mean_encode_room_type)
test.loc[:,'Neighbourhood Cleansed'] = test['Neighbourhood Cleansed'].map(mean_encode_barrio)
test.loc[:,'Neighbourhood Group Cleansed'] = test['Neighbourhood Group Cleansed'].map(mean_encode_distrito)
test.loc[:,'Cancellation Policy'] = test['Cancellation Policy'].map(mean_encode_cancellation_policy)


#Cambiamos el nombre de la variable a Barrio
test.rename(columns={'Neighbourhood Cleansed':'Barrio'}, inplace=True)
test.rename(columns={'Neighbourhood Group Cleansed':'Distrito'}, inplace=True)

# Como pueden quedar valores nulos en Barrio y Cancellation Policy debido a aplicar sobre ellas el Mean Encoding de train, 
# lo rellenaremos con la media de las medias obtenidas para esas categoria
mean_mean_encode_barrio = np.mean(mean_encode_barrio)
mean_mean_encode_distrito = np.mean(mean_encode_distrito)
mean_mean_encode_cancellation_policy = np.mean(mean_encode_cancellation_policy)
test['Barrio'].fillna(mean_mean_encode_barrio, inplace=True)
test['Distrito'].fillna(mean_mean_encode_distrito, inplace=True)
test['Cancellation Policy'].fillna(mean_mean_encode_cancellation_policy, inplace=True)


#Por ultimo guardamos estos dataset (train y test) que son los que usaremos para entrenar nuestro modelo
#fn = os.path.join(os.path.dirname(__file__), 'data/train_model.csv')
#train.to_csv(fn, sep=';', decimal='.', index=False)
train.to_csv('model/data/train_model.csv', sep=';', decimal='.', index=False)

#fn = os.path.join(os.path.dirname(__file__), 'data/test_model.csv')
#test.to_csv(fn, sep=';', decimal='.', index=False)
test.to_csv('model/data/test_model.csv', sep=';', decimal='.', index=False)

my_logger.info(f'Dimensiones del dataset test al final : {test.shape}')

