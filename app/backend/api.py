from flask import render_template

from app.backend import app_flask
from app.backend import utils as utils

##sockets = Sockets(app=app_flask)

@app_flask.route('/')
@app_flask.route('/index')
def index():
    user = {'nickname': 'unknown'}
    return render_template("index.html", title='Home', user=user)
