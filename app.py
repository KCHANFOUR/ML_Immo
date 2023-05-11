# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
from test_model import model

app = Flask(__name__)

X = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    X = [request.form['squareMeters'],request.form['numberOfRooms'],request.form['floors'],request.form['cityCode'],request.form['cityPartRange'],request.form['numPrevOwners'],request.form['made'],request.form['basement'],request.form['attic'],request.form['garage']]
    result = model(X)
    return f'Hello, {result}!'


if __name__ == '__main__':
    app.run()
