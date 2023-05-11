# -*- coding: utf-8 -*-

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    name = request.form['name']
    return f'Hello, {name}!'


if __name__ == '__main__':
    app.run()
