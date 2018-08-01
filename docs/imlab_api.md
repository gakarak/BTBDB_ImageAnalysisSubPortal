# imalab api

BTBDB Image Analysis Controller:( [image-analysis-controller-01.tbportal.org](http://image-analysis-controller-01.tbportal.org/) )

## Series request:
```
@app_flask.route('/report/<string:case_id>/<string:patient_id>/<string:study_uid>/<string:series_uid>/', methods=['GET'])
```

[Example: series response](imlab_response_series.json)


-----
## Study request
```
@app_flask.route('/study_info/<string:case_id>/<string:patient_id>/<string:study_uid>/', methods=['GET'])
```

[Example: study response](imlab_response_study.json)

