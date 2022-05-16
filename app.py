import numpy as np

from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

import pickle

app = Flask(__name__)  # initialising flask app

model = pickle.load(open('rf_random', 'rb')) # load ml model
sc = pickle.load(open('scc', 'rb')) # load ml model
ex = pickle.load(open('EX', 'rb')) # load ml model
assupred = pickle.load(open('GBT', 'rb')) # load ml model
assutype = pickle.load(open('knn', 'rb')) # load ml model
std = pickle.load(open('std', 'rb')) # load ml model

standardScaler = StandardScaler()
standardScaler.fit(sc)


@app.route('/carpred', methods=['GET'])
def home():
    return render_template('car_pred.html')


@app.route('/carpred', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        Kilometers_Driven = float(request.form['Kilometers_Driven'])
        Engine = float(request.form['Engine'])
        seats = float(request.form['seats'])
        car_age = int(request.form['age'])
        fuel_type = request.form['fuel']
        transmission_type = request.form['transmission']

        
        
        if fuel_type == 'Diesel':
            fuel_type = 1
        elif fuel_type == 'Petrol':
            fuel_type = 0
        elif fuel_type == 'CNG':
            fuel_type = 2
        elif fuel_type == 'LPG':
            fuel_type = 3
        else:
            fuel_type = 4
            

        if transmission_type == 'Manual':
            transmission_type = 1
        else:
            transmission_type = 0
        
        features = np.array
        ([[car_age,Kilometers_Driven,fuel_type,transmission_type,Engine,seats,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])

        scc=0
        print(scc)
        #model = pickle.load(open('model', 'rb'))  # load ml model
        prediction = model.predict(standardScaler.transform([[car_age,Kilometers_Driven,fuel_type,transmission_type,Engine,seats,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]]))
        output = round(prediction[0]/2.7, 2)
     #   output =ex('CIN1.jpg','13CG_F.jpg','1CG_B.jpg')
        return render_template('car_pred.html', output="{} DT".format(output))
@app.route('/assupred', methods=['GET'])
def homeassu():
    return render_template('assu_pred.html')
    
    
@app.route('/assupred', methods=['POST', 'GET'])
def assupredict():
    if request.method == 'POST':
        selling_price = float(request.form['selling_price'])
        Engine = int(request.form['Engine'])
        type_ass = request.form['type_ass']
        car_age = int(request.form['age'])

        
        
        if type_ass == 'routiere+options':
            type_ass = 0
        elif type_ass == 'dommage et collision':
            type_ass = 1
        else: 
            type_ass = 2
            
        featuresassu = np.array
        ([[selling_price,Engine,type_ass,car_age]])

        scc=0
        print(scc)
        predictionassu = assupred.predict(std.transform([[selling_price,Engine,type_ass,car_age]]))
    
       
    
      
        return render_template('assu_pred.html', predictionassu="{} DT".format(predictionassu))
@app.route('/typepred', methods=['GET'])
def hometypeassu():
    return render_template('assutype.html')
    
    
@app.route('/typepred', methods=['POST', 'GET'])
def typeassupredict():
    if request.method == 'POST':
        selling_price = float(request.form['selling_price'])
        Engine = int(request.form['Engine'])
        prix_assuran = float(request.form['prix_assuran'])
        car_age = int(request.form['age'])

        
        
       
            
        featuresassu = np.array
        ([[selling_price,Engine,prix_assuran,car_age]])

        predictiontypeassu = assutype.predict([[selling_price,Engine,prix_assuran,car_age]])
    
       
    
      
        return render_template('assutype.html', predictiontypeassu="{} ".format(predictiontypeassu))


if __name__ == '__main__':
    app.run(debug=True)
