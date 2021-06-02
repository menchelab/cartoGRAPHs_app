
from sys import platform

if (platform == "darwin"):
    filePre = ''
if (platform == "linux"):
    filePre = "/var/www/appSLC/appSLC/"
if (platform == "win32"):
    filePre = ''