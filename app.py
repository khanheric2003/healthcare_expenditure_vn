from flask import Flask,render_template,request,redirect
#from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
model = pickle.load(open(r'C:\Users\ACER\repos\week4\model\linear_reg_model.pkl','rb'))
info = pd.read_csv(r"C:\Users\ACER\repos\week4\model\clean_model.csv")

@app.route('/',methods=['GET','POST'])
def index():
    sexes = sorted(info["sex"].unique())
    urban_opt = sorted(info["urban"].unique())
    farm_opt = sorted(info["farm"].unique())

    return render_template('index.html' , sexes=sexes,urban_opt=urban_opt,farm_opt = farm_opt)



@app.route('/predict',methods=['POST'])
def predict():
    sex = request.form.get('sex')
    urban_opt = request.form.get('urban_opt')
    farm_opt = request.form.get('farm_opt')
    age = request.form.get('age')
    edu_year = request.form.get('edu_year')
    household_size = request.form.get('household_size')
    food_exp = request.form.get('food_exp')
    commune = request.form.get('commune')
    print(sex,urban_opt,farm_opt,age,edu_year,household_size,food_exp,commune)
    

    prediction=model.predict(pd.DataFrame(columns=['sex','age','educyr','farm','urban','hhsize','rlfood','commune'],
                              data=np.array([sex,age,edu_year,farm_opt,urban_opt,household_size,food_exp,commune]).reshape(1, 8)))                              
    return str(np.round(prediction[0],2))
if __name__=='__main__':
    app.run(debug=True)