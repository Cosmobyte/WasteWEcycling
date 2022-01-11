from flask import Flask, request, jsonify
from fastai.vision.all import *
from fastai.data.external import *
# import urllib.request

# MODEL_URL = "https://drive.google.com/uc?export=download&id=1kJJTB9mhzTS_xrui1eFSct96KeRNo9BY"
# urllib.request.urlretrieve(MODEL_URL, "my_export1.pkl")

# # model = load_learner("my_export1.pkl")
# model = load_learner(Path("."), "my_export1.pkl")

model = load_learner("my_export1.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def process_image():
    
    imagefile = request.files['image0']
    img_pil = PILImage.create(imagefile)
  
    pred,pred_idx,probs = model.predict(img_pil)

    return jsonify({'prediction':str(pred),'probability':str(probs[pred_idx])})
   

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get('PORT', 5000), debug=True)