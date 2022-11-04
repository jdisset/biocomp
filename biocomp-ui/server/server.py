import sys

sys.path.append('../../scripts/')
from flask_cors import CORS
from flask import Flask, request
from pathlib import Path
import pandas as pd
import json

import biocomp as bc
import scriptutils as ut

print('Loading data...')
lib = ut.load_lib()
print('Data loaded.')

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources={r"/*": {"origins": "*"}})
# CORS(app)


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/get_all')
def getAll():
    l2s = lib.L2s.reset_index()
    l2s['type'] = 'L2'
    l1s = lib.L1s.reset_index()
    l1s['type'] = 'L1'
    plasmids = pd.concat([l2s, l1s])
    # rename id as source_id
    plasmids = plasmids.rename(columns={'id': 'source_id'})

    return plasmids.to_json(orient='records')


# @app.route('/email', methods=['POST'])
# aptid = int(request.json['apt_id'])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000", debug=True, use_reloader=True)
