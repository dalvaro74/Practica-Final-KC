#!/bin/bash

#Le ponemos el & para que gunicon se lance en background (detached) y pueda lanzarse tambien mongod
exec gunicorn --reload app:app --bind 0.0.0.0:5000 &
/usr/bin/mongod --bind_ip_all

