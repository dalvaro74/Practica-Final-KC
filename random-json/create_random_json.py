'''
Vamos a crear json de manera aleatoria a partir de listas de datos

'''
import random
import numpy as np
import json
# En primer lugar cargamos las listas desde los ficheros de datos

list_barrios = open("random-json/barrios.txt").read().splitlines()

list_room_type = open("random-json/room_type.txt").read().splitlines()

list_names= open("random-json/names.txt").read().splitlines()

list_users= open("random-json/users.txt").read().splitlines()

list_isp = open("random-json/isp.txt").read().splitlines()

list_capacidad = [i for i in range(1,11)]

list_banios = [i for i in range(1,6)]

list_entero = [i for i in range(30,180)]

list_dec = [round(elem,2) for elem in np.linspace(0,1,99)]


list_banios

def generate_random_json():
    name = random.choice(list_names)
    user = random.choice(list_users)
    isp = random.choice(list_isp)
    email = user + isp
    barrio = random.choice(list_barrios)
    room_type = random.choice(list_room_type)
    capacidad = random.choice(list_capacidad)
    banios = random.choice(list_banios)
    entero = random.choice(list_entero)
    decimal = random.choice(list_dec)
    price = entero + decimal
    data_set = {
        "name":name,
        "email":email,
        "barrio":barrio,
        "room_type":room_type,
        "accomodates":capacidad,
        "bathrooms":banios,
        "price":price
    }
    json_dump = json.dumps(data_set)
    return json_dump

f = open("random-json/usuarios.zip.json", "a")
for i in range(1000):
    user = generate_random_json()
    f.write(user)
    f.write("\n")
f.close()    



