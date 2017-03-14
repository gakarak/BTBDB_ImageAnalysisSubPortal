#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import json

if __name__ == '__main__':
    fjson = 'api_eduard_crdf_v1.json'
    strJson = open(fjson, 'r').read()
    datJson = json.loads(strJson)
    print (json.dumps(datJson, indent=4))
