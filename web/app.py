from flask import Flask, jsonify, request, render_template,  redirect 
import numpy as np
import pandas as pd
import pickle
import os

#Como vamos a dockenizar esta aplicacion deberemos usar os.path
# Cargamos nuestra Red Neuronal
fn_model = os.path.join(os.path.dirname(__file__), 'pkls/nn_model.pkl')
model = pickle.load(open(fn_model, 'rb'))

#Cargamos los decoder de barrio y room_type 
fn_barrio = os.path.join(os.path.dirname(__file__), 'pkls/mean_encode_barrio.pkl')
encoder_barrio = pickle.load(open(fn_barrio, 'rb'))
fn_room_type = os.path.join(os.path.dirname(__file__), 'pkls/mean_encode_room_type.pkl')
encoder_room_type = pickle.load(open(fn_room_type, 'rb'))


#Inicializamos la Base de Datos y su collecion asociada 
from pymongo import MongoClient

#En otra maquina
#client = MongoClient("mongodb://db:27017")
# Dentro de la misma maquina
client = MongoClient("mongodb://127.0.0.1:27017")
db = client.Airbnb
Users = db["Usuarios"]


# Initialize the app and set a secret_key.
app = Flask(__name__)
app.secret_key = 'something_secret'

#Definimos las funciones que usaran nuestros endpoints
def get_price(dic_param):

    name= dic_param['name']
    email= dic_param['email']
    barrio_cat = dic_param['barrio']
    barrio= float(encoder_barrio[barrio_cat])
    room_type_cat = dic_param['room_type']
    room_type= float(encoder_room_type[room_type_cat])
    accommodates= float(dic_param['accommodates'])
    bathrooms= float(dic_param['bathrooms'])
    #cleaning_fee= float(dic_param['cleaning_fee'])
    #extra_people= float(dic_param['extra_people'])

    np_array_param = np.array([[barrio,room_type,accommodates,bathrooms]])
    #price_predict = float(model.predict(np_array_param)[0])
    price_predict = round(float(model.predict(np_array_param)[0]), 2)
    #response = dict(ESTIMATE=price_predict, MESAGE="Todo Ok")

    #Salvamos la informacion en la Base de Datos
    Users.insert_one({
    'name':name,
    'email':email,
    'barrio':barrio_cat,
    'room_type':room_type_cat,
    'accommodates':accommodates,
    'bathrooms':bathrooms,
    #'cleaning_fee':cleaning_fee,
    #'extra_people':extra_people,
    'price':price_predict
    })
    # Devolvemos el precio a la API 
    return price_predict
    


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    dic_param ={}
    dic_param['name'] = data['name']
    dic_param['email'] = data['email']
    dic_param['barrio'] = data['barrio']
    dic_param['room_type'] = data['room_type']
    dic_param['accommodates'] = data['accommodates']
    dic_param['bathrooms'] = data['bathrooms']
    #dic_param['cleaning_fee'] = data['cleaning_fee']
    #dic_param['extra_people'] = data['extra_people']

    #Lllamamos a la funcion que se encarga de predecir con el modelo nn
    price = get_price(dic_param) 
    #response_text = 'Precio aprox: <br> {}€'.format(price)
    response_text = '<div class="solucion"> <span style="margin-bottom: 5px;" >Precio Aprox:</span><br> <span>{}€</span> </div>'.format(price)
    return render_template('index.html', prediction_text=response_text)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(debug=True)