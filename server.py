from flask import Flask, request, jsonify
from predict_clothes import predict_clothes

app = Flask(__name__)


@app.route("/")
def get_clothes_prediction():
    user_file = request.files.get("file")
    prediction = predict_clothes(user_file)
    return jsonify(prediction)


# config
DEBUG = True
if __name__ == "__main__":
    app.run(debug=DEBUG)
