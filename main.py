from flask import Flask, request, jsonify
from fastbook import load_learner
from fastai.vision.all import *
from fastai.data.external import *

import pickle



model = load_learner("my_export1.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def process_image():
    file = request.files['image0']
    # Read the image via file.stream
    
    img_pil = PILImage.create(file)
  
    pred,pred_idx,probs = model.predict(img_pil)

    return jsonify({'prediction':str(pred),'probability':str(probs[pred_idx])})


if __name__ == "__main__":
    app.run(debug=True)
