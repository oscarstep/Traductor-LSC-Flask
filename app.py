import flask
from kerasmodel import prediction
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
        data = request.get_json(force=True)
        letter = prediction(data)
        print(letter)
    return jsonify({'letra': letter})

app.run()
