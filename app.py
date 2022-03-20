from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("front.html")



@app.route("/submit", methods=['POST'])
def submit():
    if request.method == "POST":
        EMPLOYER = request.form['EMPLOYER']
        NAICS_CODE = request.form['NAICS_CODE']
        SOC_TITLE2 = request.form['SOC_TITLE2']
        SECONDARY_ENTITY = request.form['SECONDARY_ENTITY']
        AGENT_REPRESENTING_EMPLOYER = request.form['AGENT_REPRESENTING_EMPLOYER']
        CONTINUED_EMPLOYMENT = request.form['CONTINUED_EMPLOYMENT']
        CHANGE_PREVIOUS_EMPLOYMENT = request.form['CHANGE_PREVIOUS_EMPLOYMENT']
        NEW_CONCURRENT_EMPLOYMENT = request.form['NEW_CONCURRENT_EMPLOYMENT']
        CHANGE_EMPLOYER = request.form['CHANGE_EMPLOYER']
        AMENDED_PETITION = request.form['AMENDED_PETITION']
        H_1B_DEPENDENT = request.form['H-1B_DEPENDENT']
        SUPPORT_H1B = request.form['SUPPORT_H1B']
        WILLFUL_VIOLATOR = request.form['WILLFUL_VIOLATOR']
        WAGE_RATE_OF_PAY_FROM_1 = request.form['WAGE_RATE_OF_PAY_FROM_1']
        WAGE_UNIT_OF_PAY_1 = request.form['WAGE_UNIT_OF_PAY_1']
        TOTAL_WORKER_POSITIONS = request.form['TOTAL_WORKER_POSITIONS']
        FULL_TIME_POSITION = request.form['FULL_TIME_POSITION']
        array = []
        array.append(EMPLOYER)
        array.append(NAICS_CODE)
        array.append(SOC_TITLE2)
        array.append(SECONDARY_ENTITY)
        array.append(AGENT_REPRESENTING_EMPLOYER)
        array.append(CONTINUED_EMPLOYMENT)
        array.append(CHANGE_PREVIOUS_EMPLOYMENT)
        array.append(NEW_CONCURRENT_EMPLOYMENT)
        array.append(CHANGE_EMPLOYER)
        array.append(AMENDED_PETITION)
        array.append(H_1B_DEPENDENT)
        array.append(SUPPORT_H1B)
        array.append(WILLFUL_VIOLATOR)
        array.append(WAGE_RATE_OF_PAY_FROM_1)
        array.append(WAGE_UNIT_OF_PAY_1)
        array.append(TOTAL_WORKER_POSITIONS)
        array.append(FULL_TIME_POSITION)
        new_array = np.array(array)
        new_array = new_array.reshape(1,-1)
        final_result = ml_model(new_array)
        return(final_result) 

def ml_model(X_test):
    modelPath = 'model1.sav'
    nb_model = pickle.load(open(modelPath, 'rb')) 
    pred = nb_model.predict(X_test)
    if pred == 1 :
        result = "Accepted"
    if pred == 0 :
        result = "Denied" 
    return(result)

      

if __name__ == "__main__":
    app.run(debug=True)
    