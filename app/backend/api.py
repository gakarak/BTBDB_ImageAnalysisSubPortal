from flask import render_template

from flask import request, Response

from app.backend import app_flask
from app.backend import utils as utils
from app.core.dataentry_v1 import DBWatcher

import json

##########################################
def dir_base():
    return app_flask.config['DIR_BASE']

def dir_data():
    return app_flask.config['DIR_DATA']

def get_responce(retcode=0, errorstr='', result=None):
    return {
        'retcode': retcode,
        'errorstr': errorstr,
        'result': result
    }

def get_series_responce(series):
    return {
        'case_id':    series.ptrCase.caseId(),
        'patient_id': series.ptrCase.patientId(),
        'study_uid':  series.studyUID(),
        'series_uid': series.uid()
    }

##########################################
dbWatcher = DBWatcher(pdir=dir_data())
print (dbWatcher.toString())

##sockets = Sockets(app=app_flask)

##########################################
@app_flask.route('/')
@app_flask.route('/index/')
def index():
    user = {'nickname': 'unknown'}
    return render_template("index.html", title='Home', user=user)

@app_flask.route('/db/')
@app_flask.route('/db/status/')
def db_status():
    try:
        ret = get_responce(result=dbWatcher.getStatistics())
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-status *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/')
def db_series():
    try:
        ret = dbWatcher.getStatistics()
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series *error: {0}'.format(err))
    return Response(json.dumps(ret['series'], indent=4), mimetype='application/json')

@app_flask.route('/db/series/all')
def db_series_all():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-all *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/good')
def db_series_good():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isGood():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-good *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/downloaded')
def db_series_downloaded():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isDownloaded():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-downloaded *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/converted')
def db_series_converted():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isConverted():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-downloaded *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/postprocessed')
def db_series_postprocessed():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isPostprocessed():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-downloaded *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

