from flask import render_template

from flask import request, Response, make_response
from flask import send_file

import os
from app.backend import app_flask
from app.backend import utils as utils
from app.core.dataentry_v1 import DBWatcher

from xhtml2pdf import pisa
from cStringIO import StringIO

import json

##########################################
def dir_base():
    return app_flask.config['DIR_BASE']

def dir_data():
    return app_flask.config['DIR_DATA']

def get_response(retcode=0, errorstr='', result=None):
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
def report_helper(case_id, patient_id, study_uid, series_uid, root_url, url_basic=""):
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
                #FIXME: temporary solution
                ret['pdf_report'] = {
                    'url': '{4}report-pdf/{0}/{1}/{2}/{3}'.format(case_id, patient_id, study_uid, series_uid, url_basic),
                    'description': 'No comments...'
                }
                return Response(json.dumps(get_response(result=ret), indent=4), mimetype='application/json')
            except Exception as err:
                str_error = 'db-report error: {0}'.format(err)
    else:
        str_error = 'Cant find requested series in DB: case_id={0}, study_uid={1}, series_uid={2}' \
            .format(case_id, study_uid, series_uid)
    if is_ok:
        return Response(json.dumps(get_response(result=ret), indent=4), mimetype='application/json')
    else:
        return Response(json.dumps(get_response(1, str_error), indent=4), mimetype='application/json')

##########################################
def cases_info(isCheckProcessed=True):
    responseCases = []
    # arrAges = []
    # arrGender = []
    for kcase, case in dbWatcher.cases.items():
        caseId = case.caseId()
        retAge = case.dictShort['ageOnset']
        retDiag = case.dictShort['diagnosis']['display']
        retGender = case.dictShort['patientGender']
        # try:
            # arrAges.append(int(retAge))
            # arrGender.append()
        # except:
        #     pass
        arrSeries = []
        if case.isInitialized() and (not case.isEmpty()):
            for kseries, series in case.series.items():
                if (not isCheckProcessed) or (isCheckProcessed and series.isPostprocessed()):
                    retSeries = {
                        'case_id': case.caseId(),
                        'patient_id': case.patientId(),
                        'study_uid': series.studyUID(),
                        'series_uid': series.uid()
                    }
                    arrSeries.append(retSeries)
        if len(arrSeries) > 0:
            retCase = {
                'case_id': caseId,
                'age': retAge,
                'diag': retDiag,
                'gender': retGender,
                'series': arrSeries
            }
            responseCases.append(retCase)
    return responseCases

##########################################
@app_flask.route('/data/', methods=['GET'])
def data_load():
    file_path = request.args.get('path')
    # file_path = os.path.join(dir_data(), file_path)
    file_path = '{0}/{1}'.format(dir_data(), file_path)
    if os.path.isfile(file_path):
        return send_file(file_path, mimetype='image/jpeg')
        # with open(file_path, 'r') as f:
        #     return f.read()
    return Response(json.dumps({}), mimetype='application/json')

@app_flask.route('/report-pdf/<string:case_id>/<string:patient_id>/<string:study_uid>/<string:series_uid>/', methods=['GET'])
def data_pdf_load(case_id, patient_id, study_uid, series_uid):
    try:
        jsonResponse = report_helper(case_id=case_id, patient_id=None, study_uid=study_uid, series_uid=series_uid, root_url="")
        retJson = json.loads(jsonResponse.get_data())
        retJson = retJson['responce']
        tmpImgPath = retJson['preview_images'][0]['url']
        retJson['preview_images'][0]['url'] = '{0}/{1}'.format(dir_data(), tmpImgPath)
        strHTML = render_template('templates/template_pdf.html', dataJson = retJson)
        pdf = StringIO()
        pisa.CreatePDF(StringIO(strHTML), pdf)
        retResponse = make_response(pdf.getvalue())
        retResponse.headers['Content-Sispositions'] = "attachment; filename='crdf-report.pdf'"
        retResponse.mimetype = 'application/pdf'
        return retResponse
    except Exception as err:
        str_error = 'db-report error: {0}'.format(err)
        return Response(json.dumps(get_response(1, str_error), indent=4), mimetype='application/json')

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
        ret = get_response(1, str_error)
        return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/report/<string:case_id>/<string:patient_id>/<string:study_uid>/<string:series_uid>/', methods=['GET'])
def report_path(case_id, patient_id, study_uid, series_uid):
    root_URL = request.url_root
    return report_helper(case_id, patient_id, study_uid, series_uid, '{0}data/?path='.format(root_URL), url_basic=root_URL)

##########################################
# db-info REST API
@app_flask.route('/db/', methods=['GET'])
@app_flask.route('/db/status/', methods=['GET'])
def db_status():
    try:
        ret = get_response(result=dbWatcher.getStatistics())
    except Exception as err:
        ret = get_response(retcode=1, errorstr='db-status *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/cases-info/', methods=['GET'])
def case_info():
    try:
        ret = get_response(result=cases_info(isCheckProcessed=True))
    except Exception as err:
        ret = get_response(retcode=1, errorstr='cases-info *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/', methods=['GET'])
def db_series():
    try:
        ret = dbWatcher.getStatistics()
    except Exception as err:
        ret = get_response(retcode=1, errorstr='db-series *error: {0}'.format(err))
    return Response(json.dumps(ret['series'], indent=4), mimetype='application/json')

@app_flask.route('/db/series/all/', methods=['GET'])
def db_series_all():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized():
                ret.append(get_series_responce(ser))
    except Exception as err:
        ret = get_response(retcode=1, errorstr='db-series-all *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/good/', methods=['GET'])
def db_series_good():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isGood():
                ret.append(get_series_responce(ser))
        ret = get_response(result=ret)
    except Exception as err:
        ret = get_response(retcode=1, errorstr='db-series-good *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/downloaded/', methods=['GET'])
def db_series_downloaded():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isDownloaded():
                ret.append(get_series_responce(ser))
        ret = get_response(result=ret)
    except Exception as err:
        ret = get_response(retcode=1, errorstr='db-series-downloaded *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/converted/', methods=['GET'])
def db_series_converted():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isConverted():
                ret.append(get_series_responce(ser))
        ret = get_response(result=ret)
    except Exception as err:
        ret = get_response(retcode=1, errorstr='db-series-downloaded *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

@app_flask.route('/db/series/processed/', methods=['GET'])
def db_series_postprocessed():
    try:
        ret = []
        for ser in dbWatcher.allSeries():
            if ser.isInitialized() and ser.isPostprocessed():
                ret.append(get_series_responce(ser))
        ret = get_response(result=ret)
    except Exception as err:
        ret = get_response(retcode=1, errorstr='db-series-downloaded *error: {0}'.format(err))
    return Response(json.dumps(ret, indent=4), mimetype='application/json')

