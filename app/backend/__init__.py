import os

from flask import Flask
from config import DevelopmentConfig

app_flask = Flask(__name__, static_folder='../frontend', template_folder='../frontend')

config = DevelopmentConfig()
app_flask.config.from_object('config.DevelopmentConfig')

if app_flask.config['DEBUG']:
    print('** DIR_BASE = {0}'.format(app_flask.config['DIR_BASE']))
    print('** DIR_DATA = {0}'.format(app_flask.config['DIR_DATA']))

from app.backend import api
