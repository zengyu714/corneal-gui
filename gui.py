import os
from random import randint

from flask import Flask, flash, redirect, render_template, request, session, abort
from flask import send_from_directory

from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/images'), 'favicon.ico', mimetype='image/x-icon')


@app.route('/')
def index():
    return render_template('apperance.html', **locals())


if __name__ == '__main__':
    app.run(host="172.21.20.65", port=5001)
