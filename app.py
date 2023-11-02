import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# Create flask app
app = Flask(__name__ , template_folder="template")
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features)
    return render_template("index.html", prediction_text = "PREDICTION STRENGTH OF CEMENT MIXTURE : {}".format(prediction))

if __name__=="__main__":
    app.run(host='0.0.0.0',port = '5000')
