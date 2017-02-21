#!/bin/bash

wdir=`dirname $0`

cd $wdir

export PYTHONPATH=$PYTHONPATH:$PWD/app/backend

python run-app.py


