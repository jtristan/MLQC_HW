import requests
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from rdkit import Chem
import mordred
from mordred import Calculator, descriptors

scoreboard = 'ffirhduscve'

get = 'https://keepthescore.co/api/' + scoreboard + '/board'
post = 'https://keepthescore.co/api/' + scoreboard + '/add_single_score'

def report(teamname,score):
    test_error = score
    r = requests.get(get)
    scores = json.loads(r.text)
    for entry in scores['players']:
        if entry['player_name'] == teamname:
            diff = test_error - entry['score']
            break
    if (diff < 0):
        commit(diff,teamname)
    else:
        print('Your score is worse than your previous best score, it will not be reported.')

def commit(diff,teamname):
    requests.post(post, json={ "player_name": teamname, "score": diff, "comment": "test" })

def test(predictor,teamname):
    data_E_E0 = pd.read_csv('dataset-E-E0.csv')
    E_X = np.array(data_E_E0['SMILES'])
    E_E0 = np.array(data_E_E0['E0'])
    y_pred = np.zeros(E_E0.shape)
    for i in range(len(E_X)):
        y_pred[i] = predictor(E_X[i])[0]
    mae = MAE(y_pred,E_E0) * 23.06
    report(teamname,mae)
    return mae 
