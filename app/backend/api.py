from flask import render_template

from flask import request, Response

import os
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
        'responce': result
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

##########################################
def report_helper(case_id, patient_id, study_uid, series_uid, root_url):
    ptr_series = None
    is_ok = False
    ret = None
    for ser in dbWatcher.allSeries():
        if ser.isInitialized() \
                and (case_id == ser.caseId()) \
                and (study_uid == ser.studyUID()) \
                and (series_uid == ser.uid()):
            ptr_series = ser
    if ptr_series is not None:
        is_ok = True
        str_error = ''
        if not ptr_series.isDownloaded():
            is_ok = False
            str_error = 'data for requested series has not yet been downloaded'
        if not ptr_series.isConverted():
            is_ok = False
            str_error = 'DICOM data for requested series downloaded, but has not yet been converted to Nifti'
        if not ptr_series.isPostprocessed():
            is_ok = False
            str_error = 'data downloaded and converted but has not been processed yet...'
        if is_ok:
            try:
                ret = ptr_series.getReportJson(root_url = root_url)
                return Response(json.dumps(get_responce(result=ret), indent=4), mimetype='application/json')
            except Exception as err:
                str_error = 'db-report error: {0}'.format(err)
    else:
        str_error = 'Cant find requested series in DB: case_id={0}, study_uid={1}, series_uid={2}' \
            .format(case_id, study_uid, series_uid)
    if is_ok:
        return Response(json.dumps(get_responce(result=ret), indent=4), mimetype='application/json')
    else:
        return Response(json.dumps(get_responce(1, str_error), indent=4), mimetype='application/json')

@app_flask.route('/data/', methods=['GET'])
def data_load():
    file_path = request.args.get('path')
    file_path = os.path.join(dir_data(), file_path)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            return f.read()
    return Response(json.dumps({}), mimetype='application/json')

@app_flask.route('/report/', methods=['POST'])
def report_json():
    try:
        case_id = request.args.get('case_id')
        patient_id = request.args.get('patient_id')
        study_uid = request.args.get('study_uid')
        series_uid = request.args.get('series_uid')
        return report_helper(case_id, patient_id, study_uid, series_uid)
    except Exception as err:
        str_error = 'Invalid request: [{0}]'.format(err)
        ret = get_responce(1, str_error)
        return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/report/<string:case_id>/<string:patient_id>/<string:study_uid>/<string:series_uid>/', methods=['GET'])
def report_path(case_id, patient_id, study_uid, series_uid):
    root_URL = request.url_root
    return report_helper(case_id, patient_id, study_uid, series_uid, '{0}data/?path='.format(root_URL))

    # ptr_series = None
    # for ser in dbWatcher.allSeries():
    #     if ser.isInitialized() \
    #             and (case_id == ser.caseId()) \
    #             and (study_uid == ser.studyUID()) \
    #             and (series_uid == ser.uid()):
    #         ptr_series = ser
    # if ptr_series is not None:
    #     is_ok = True
    #     str_error = ''
    #     if not ptr_series.isDownloaded():
    #         is_ok = False
    #         str_error = 'data for requested series has not yet been downloaded'
    #     if not ptr_series.isConverted():
    #         is_ok = False
    #         str_error = 'DICOM data for requested series downloaded, but has not yet been converted to Nifti'
    #     if not ptr_series.isPostprocessed():
    #         is_ok = False
    #         str_error = 'data downloaded and converted but has not been processed yet...'
    #     if is_ok:
    #         try:
    #             ret = ptr_series.getReportJson()
    #             return Response(json.dumps(get_responce(result=ret), indent=4), mimetype='application/json')
    #         except Exception as err:
    #             str_error = 'db-report error: {0}'.format(err)
    #     return Response(json.dumps(get_responce(1, str_error), indent=4), mimetype='application/json')
    # else:
    #     str_error = 'Cant find requested series in DB: case_id={0}, study_uid={1}, series_uid={2}' \
    #         .format(case_id, study_uid, series_uid)
    #     ret = get_responce(retcode=1, errorstr=str_error)
    #     return Response(json.dumps(ret, indent=4), mimetype='application/json')

##########################################
# db-info REST API
@app_flask.route('/db/', methods=['GET'])
@app_flask.route('/db/status/', methods=['GET'])
def db_status():
    try:
        ret = get_responce(result=dbWatcher.getStatistics())
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-status *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/', methods=['GET'])
def db_series():
    try:
        ret = dbWatcher.getStatistics()
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series *error: {0}'.format(err))
    return Response(json.dumps(ret['series'], indent=4), mimetype='application/json')

@app_flask.route('/db/series/all/', methods=['GET'])
def db_series_all():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-all *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/good/', methods=['GET'])
def db_series_good():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isGood():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-good *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/downloaded/', methods=['GET'])
def db_series_downloaded():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isDownloaded():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-downloaded *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/converted/', methods=['GET'])
def db_series_converted():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isConverted():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-downloaded *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/processed/', methods=['GET'])
def db_series_postprocessed():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isPostprocessed():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_responce(retcode=1, errorstr='db-series-downloaded *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

