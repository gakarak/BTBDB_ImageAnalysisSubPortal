#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from jinja2 import Template
import xhtml2pdf.pisa as pisa
import json
import cStringIO



def_ = 'inp_data'

if __name__ == '__main__':
    fnTmpl = 'template_pdf.html'
    foutPDF = 'render.pdf'
    wdir = 'inp_data'
    fnReportJson = '{0}/report.json'.format(wdir)
    fnReportImgs = '{0}/preview0.jpg'.format(wdir)
    strTmpl = open(fnTmpl, 'r').read()
    strJson = open(fnReportJson, 'r').read()
    dataJson = json.loads(strJson)
    dataJson['preview_images'][0]['url'] = fnReportImgs
    #
    template = Template(strTmpl)
    dataHTML = template.render(dataJson=dataJson)
    print (dataHTML)
    #
    pdf = pisa.CreatePDF(
        cStringIO.StringIO(dataHTML),
        file(foutPDF, "wb")
    )

