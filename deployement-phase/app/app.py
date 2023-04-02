import io
import json

import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import psycopg2


app = Flask(__name__)


imagenet_class_index = json.load(open('image_class_index.json'))

model = torch.jit.load('model1.pt')
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        data = {'class_id': class_id, 'class_name': class_name}
        write_db(data)
        return jsonify(data)
    

def write_db(data):
    db_conn = psycopg2.connect(
        host="postgres",
        port="5432",
        database="postgres",
        user="postgres",
        password="postgres")
    
    cur = db_conn.cursor()
    query = f"""
    INSERT INTO model (class_id, class_name) VALUES (%s, %s)
    """    
    values = (data['class_id'], data['class_name'])
    cur.execute(query, values)
    db_conn.commit()
    db_conn.close()
    return print("Data added to DB")

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("5000"),
        debug=True
    )