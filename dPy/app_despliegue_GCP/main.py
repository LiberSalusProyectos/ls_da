from flask import Flask, render_template, request, redirect, url_for
from sklearn.externals import joblib
import pandas as pd
import numpy as np

#app = Flask('stoke_pricer')
app = Flask(__name__)

def switch_eustreason(argument):
    switcher = {
        "Precio": 1,
        "Ubicacion": 2,
        "Calidad de los productos": 3,
        "Variedad de productos": 4,
        "Servicio": 5,
        "Otro": 6
    }
    return switcher.get(argument, 0)

def switch_eugenhth(argument):
    switcher = {
        "Excelente": 1,
        "Muy buena": 2,
        "Buena": 3,
        "Regular": 4,
        "Deficiente": 5
    }
    return switcher.get(argument, 0)

@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')

@app.route('/resultsform', methods=['POST', 'GET'])
def results():
    form = request.form
    if request.method == 'POST':
       #model = joblib.load('G:/Mi unidad/Desarrollo/desarrolloPython/ML1Obesidad_RF.pkl')
       model = joblib.load('ML1Obesidad_RF.pkl')
       inp = pd.DataFrame(columns=['euexfreq', 'eustreason', 'eugenhth', 'ertseat', 'eufastfdfrq'])
       inp = inp.append({'euexfreq': request.form['euexfreq']}, ignore_index=True)
       inp['eustreason'] = inp['eustreason'].replace({np.nan: switch_eustreason(request.form['eustreason'])})
       inp['eugenhth'] = inp['eugenhth'].replace({np.nan: switch_eugenhth(request.form['eugenhth'])})
       inp['ertseat'] = inp['ertseat'].replace({np.nan: request.form['ertseat']})
       inp['eufastfdfrq'] = inp['eufastfdfrq'].replace({np.nan: request.form['eufastfdfrq']})
       predicted_stock_price = model.predict_proba(inp)[0][1]
       return render_template('resultsform.html', inp=inp,   predicted_price=predicted_stock_price)
    else:
        return render_template('resultsform.html', predicted_price=request.method)

if __name__ == '__main__':
    app.run()
