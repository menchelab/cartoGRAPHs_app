#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr, level=1)
sys.path.insert(0,"/var/www/cartoGRAPHs_app/cartoGRAPHs_app/")

print('CSDEBUG: path inserted successfully')

from app import myServer as application
#application.secret_key = 'Add your secret key'

print('CSDEBUG: wsgi run')
