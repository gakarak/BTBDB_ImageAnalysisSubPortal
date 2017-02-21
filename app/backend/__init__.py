import os

from flask import Flask

app_flask = Flask(__name__, static_folder='../frontend', template_folder='../frontend')

from app.backend import api
