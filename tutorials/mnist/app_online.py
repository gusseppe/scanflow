import gradio as gr
import pandas as pd
import numpy as np
import json
import sys
import os
import io
import base64
import requests
from PIL import Image
sys.path.insert(0,'../..')

model_name = sys.argv[1]
model_version = int(sys.argv[2])

def inference(image):
    img = Image.fromarray(image).convert("L")

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    image_str = base64.b64encode(img_byte_arr).decode()
    
    predictor_api = 'http://localhost:8011/call'

    content = {'image_file': image_str}

    response = requests.post(
        url=predictor_api,
        data = json.dumps(content)
    )    
    result = json.loads(response.text)
    
    return result['prediction']
    
def classify(image):

    image = image[:, :, 0] #reduce the dimension to 2d
    prediction = inference(image)

    return  prediction

iface = gr.Interface(
    classify, 
    gr.inputs.Image(), 
    gr.outputs.Label(),
    capture_session=True,
#     interpretation="default",
    examples=[
        ["images/0/0-0.png"],
        ["images/1/1-1.png"],
        ["images/6/6-0.png"],
        ["images/7/7-1.png"],
        ["images/8/8-1.png"]
    ],
    server_port=7861
)

iface.launch(inline=False, debug=True) # Change inline=False in script.py