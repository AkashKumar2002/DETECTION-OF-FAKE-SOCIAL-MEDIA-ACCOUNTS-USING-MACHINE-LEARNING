#Importing necessary libraries 
import numpy as np
import pandas as pd
from flask import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

# @app.route('/load',methods=["GET","POST"])
# def load():
#     global df, dataset
#     if request.method == "POST":
#         data = request.files['data']
#         df = pd.read_csv(data)
#         dataset = df.head(100)
#         msg = 'Data Loaded Successfully'
#         return render_template('load.html', msg=msg)
#     return render_template('load.html')

@app.route('/view')
def view():
    global df, dataset
    df = pd.read_csv('data.csv')
    dataset = df.head(100)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
        data = pd.read_csv('data.csv')
        data.head()
        x=data.iloc[:,:-1]
        y=data.iloc[:,-1]

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)

        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            from sklearn.linear_model import   LogisticRegression
            lr = LogisticRegression()
            lr=lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            acc_lr = accuracy_score(y_test,y_pred)*100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Logistic Regression is ' + str(acc_lr) + str('%')
            return render_template('model.html', msg=msg)
        elif s==2:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier()
            rf=rf.fit(x_train,y_train)
            y_pred = rf.predict(x_test)
            acc_rf = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(acc_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s==4:
            dt = DecisionTreeClassifier()
            dt=dt.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            acc_dt = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(acc_dt) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')

import pickle
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = float(request.form['text'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        f8 = float(request.form['f8'])
        f9 = float(request.form['f9'])
        f10 = float(request.form['f10'])
        f11 = float(request.form['f11'])

        print(f1)

        li = [[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11]]
        print(li)
        
        filename='Random_forest.sav'
        model = pickle.load(open(filename, 'rb'))

        result =model.predict(li)
        result=result[0]
        print(result)
        if result==0:
            msg = 'The account is Genuine'
        elif result==1:
            msg= 'This is a fake account'
               
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')



if __name__ =='__main__':
    app.run(debug=True)