import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# 모델 파일 경로
MODEL_PATH = os.path.join("model", "mission_model.h5")
# 모델 로드
model = tf.keras.models.load_model(MODEL_PATH)


class_labels = ["드라켄", "석가탑", "황남빵"]  

@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    
    file = request.files.get('image')
    if not file:
        return "No file uploaded", 400

    
    mission_str = request.form.get('mission') 
    if mission_str is None:
        
        mission_idx = None
    else:
        mission_idx = int(mission_str)

    
    img = Image.open(file.stream).convert("RGB").resize((224, 224))

    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)

    
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_label = class_labels[predicted_idx]

    
    
    result_text = "미션 판별 불가(미션 선택X)"  # 기본값
    if mission_idx is not None:
        if predicted_idx == mission_idx:
            result_text = "미션 성공!"
        else:
            result_text = "미션 실패!"

    
    return render_template('result.html',
                           predicted_label=predicted_label,
                           confidence=confidence,
                           result_text=result_text,
                           mission_idx=mission_idx)


@app.route('/api/upload', methods=['POST'])
def api_upload():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    
    img = Image.open(file.stream).convert("RGB").resize((224, 224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0]))
    label = class_labels[idx]

    return jsonify({
        "predicted_label": label,
        "confidence": conf
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
