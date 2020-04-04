# Las imagenes de python vienen por defecto instaladas en buster
FROM       python:3.7

# Copy our application code
WORKDIR /var/app
COPY . .


# Installation:
RUN apt-get update
# Por si acaso para la instalacion de Mongo 
RUN apt-get install gnupg

# Fetch app specific dependencies
RUN pip3 install pip -U
#Para evitar los errores de google-auth y tensorboard
RUN pip install setuptools -U
#Sin cache
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install -r requirements.txt

#INSTALACION DE MONGODB

# Import the public key used by the package management system
RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | apt-key add -

# Create list file for MongoDB
RUN echo "deb http://repo.mongodb.org/apt/debian buster/mongodb-org/4.2 main" | tee /etc/apt/sources.list.d/mongodb-org-4.2.list

# Reload local package database
RUN apt-get update

# Install the MongoDB packages
RUN apt-get install -y mongodb-org

# Create the default data directory
RUN mkdir -p /data/db

# Define mountable directories.
VOLUME ["/data/db"]

# Expose port #27017 from the container to the host
EXPOSE 27017 5000

# Start the app
ADD start.sh /
RUN chmod +x /start.sh

CMD ["./start.sh"]
#CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
#CMD ["/usr/bin/mongod"]