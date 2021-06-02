
from sys import platform

if (platform == "darwin"):
    filePre = ''
if (platform == "linux"):
    filePre = "/var/www/cartoGRAPHs_app/cartpGRAPHs_app/"
if (platform == "win32"):
    filePre = ''