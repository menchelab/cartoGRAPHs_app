#!/usr/bin/python
import sys
import logging
logging.basicConfig(level=1, stream=sys.stderr)
sys.path.insert(0,"/var/www/cartoGRAPHs_app/")

print('path inserted successfully')

from cartoGRAPHs_app import myServer as application
#application.secret_key = 'Add your secret key'

print('wsgi run')
