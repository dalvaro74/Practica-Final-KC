'''
En este colab se llevara a cabo el proceso para la descarga de imagenes del dataset de airbnb. 
Utilizamos el campo "Thumbnail Url" que proporciona imagenes de una tamaño adecuado a nuestras necesidades (216x144).

Tambien se generara un dataset nuevo (airbnb-images-clean.csv) con aquellos registros que tengan imagenes asociadas 
con el tamaño indicado arriba, borrandose el resto.

Este dataset contendra un nuevo campo llamado image_path que contendra el el path relativo de cada imagen descargada.

El nombre de las imagenes estara asociado al campo ID de cada registro

'''

#Establecemos los imports necesarios
import numpy as np
import pandas as pd
import urllib.request
import sys
from os.path import join
import logging

# Initialize logging
logging.basicConfig(filename="logs/download_images.log",format='%(asctime)s %(message)s')
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)

my_logger.info('\n******************** COMIENZA PROCESO DE DESCARGA DE IMAGENES ********************')

# Definimos las funciones necesarias
def download_image(id,url,file_path):    
  try:
    filename = 'image-{}.jpg'.format(id)
    #full_path = '{}{}'.format(file_path, filename)
    full_path = join(file_path, filename)
    urllib.request.urlretrieve(url, full_path)
    return full_path
  except:
      my_logger.error(sys.exc_info())
      return 404


#Cargamos el fichero inicial
full_airbnb = pd.read_csv('model/data/airbnb-listings.csv',sep=';', decimal='.')

#Incluimos una nueva columna con NaN donde ira el path de las imagenes descargadas
full_airbnb["image_path"] = np.nan

#Tratamiento de las url y descarga de imagenes
from time import time
#Vamos a poner tres contadores para ver cuantas imagenes han ido bien, cuantas no tenian url (NaN) y cuantas han dado error 404
#Tambien incluimos uno global
cont_images_ok = 0
cont_urls_nan = 0
cont_urls_404 = 0
cont_global = 0

#Creamos 3 listas donde almacenaremos los IDs para cada caso
ids_image_ok = []
ids_url_nan = []
ids_url_404 = []

#Vamos a visualizar cuanto tarda el proceso
start_time = time()

for index, row in full_airbnb.iterrows():
  cont_global +=1
  if cont_global%100 == 0:
    my_logger.info('LLevamos {} Registros tratados'.format(cont_global))
  if type(row['Thumbnail Url'])==str:
    id = row['ID']
    url = row['Thumbnail Url']
    image_path = download_image(id,url,'model/images')
    if image_path == 404:
      cont_urls_404 +=1
      ids_url_404.append(id)
    else:
      cont_images_ok +=1
      ids_image_ok.append(id)
      full_airbnb.loc[index,"image_path"] = image_path
  else:
    cont_urls_nan +=1
    ids_url_nan.append(id)

elapsed_time = time() - start_time
my_logger.info("Elapsed time: %0.10f seconds." % elapsed_time)
my_logger.info('Total Registros tratados: {}'.format(cont_global))
my_logger.info('Se han subido {} imagenes correctamente'.format(cont_images_ok))
my_logger.info('Registros sin URL(NaN): {}'.format(cont_urls_nan))
my_logger.info('Registros con error 404: {}'.format(cont_urls_404))

'''
Eliminamos los registros sin imagen asociada, Ya sea porque no tienen URL en el campo "Thumbnail",
porque dicha URL no es valida (Error 404) o porque la imagen no se corresponde con el tamaño de 216x144.

Tambien se creara un nuevo dataset llamado airbnb-images.csv
'''
full_airbnb_images = full_airbnb.drop(full_airbnb[full_airbnb['image_path'].isnull()].index)
#Guardamos este dataframe como un cvs
full_airbnb_images.to_csv('model/data/airbnb-images.csv', sep=';', decimal='.', index=False)

from PIL import Image
from time import time

def get_num_pixels(filepath):
  width, height = Image.open(filepath).size
  return (width,height)
# Creamos una lista para guardar las imagenes que no tiene el tamaño standar de 216x144
image_other_size = []

cont_global=0
start_time = time()
#Asi items es un vector con el index
#for items in full_airbnb['image_path'].iteritems():
#Asi items solo contiene el valor
for items in full_airbnb_images['image_path']:
  cont_global +=1
  if cont_global%100 == 0:
    my_logger.info('LLevamos {} Registros tratados'.format(cont_global))
  if not pd.isnull(items):
    image_size = get_num_pixels(items)
    if image_size != (216, 144):
      image_other_size.append(items)

elapsed_time = time() - start_time
my_logger.info(f"Elapsed time: {elapsed_time} seconds.")
my_logger.info('Imagenes con tamanio distinto al establecido (216x144): {}'.format(len(image_other_size)))

total_imagenes_cargadas = cont_images_ok - len(image_other_size)
my_logger.info('TOTAL IMAGENES OK DATASET : {}'.total_imagenes_cargadas)

#Eliminamos los registros de esas imagenes
full_airbnb_images_clean = full_airbnb_images.drop(full_airbnb_images[full_airbnb_images['image_path'].isin(image_other_size)].index)

#Una vez limpiado el dataset con las imagenes que vamos a tratar lo guardamos en un fichero.
full_airbnb_images_clean.to_csv('model/data/airbnb-images-clean.csv', sep=';', decimal='.', index=False)
