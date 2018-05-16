import os
import sys
from app.backend import app_flask
import argparse

def runapp(pargs):
    if (pargs.port is None) or (pargs.host is None):
        p_port = 7777
        p_host = '0.0.0.0'
    else:
        p_port = pargs.port
        p_host = pargs.host
    print ('Go to URL: http://{}:{}'.format(p_host, p_port))
    app_flask.run(host=p_host, port=p_port, debug=True)
    # socketio.run(app_flask, port=port, debug=False, host='0.0.0.0')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7777, required=False, help='service port')
    parser.add_argument('--host', type=str, default='0.0.0.0', required=False, help='server host')
    args = parser.parse_args()
    #
    print('params: [{}]'.format(parser))
    runapp(args)
