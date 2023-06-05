# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import numpy as np

import test_model

app = Flask(__name__)

X = []
predicted_price = 0

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    X = [request.form['squareMeters'],request.form['numberOfRooms'],request.form['floors'],request.form['cityCode'],request.form['cityPartRange'],request.form['numPrevOwners'],request.form['made'],request.form['basement'],request.form['attic'],request.form['garage']]
    data = np.array(X,dtype=int)
    data = data.reshape(1,-1)
    predicted_price = test_model.model(data)
    print(data)
    return f'THERESULT PREDICTED BY APP: {predicted_price[0]} and the data are : {data} €'


if __name__ == '__main__':
    app.run()
