import gradio as gr
import pandas as pd
import numpy as np
import json
import sys
import os
import requests
sys.path.insert(0,'../..')

model_name = sys.argv[1]
model_version = int(sys.argv[2])


base_path = os.path.dirname(os.path.dirname(os.getcwd()))
app_dir = os.path.join(base_path, "examples/demo_mnist/")

scanflow_uri = 'http://localhost:8050/run/executor'
app_dir = '/home/guess/Desktop/scanflow/examples/demo_mnist/'
filename = 'x_inference_dashboard.npy'

content = {'name': 'inference-mnist-inference',
          'parameters':{'model_name':model_name, 
                        'model_version': model_version,
#                         'model_stage':"Production",
                        'x_inference_path': filename}}

def inference(x_inference):
    
    x_inference_path = os.path.join(app_dir, 'workflow',
                                filename)
    y_inference_path = os.path.join(app_dir, 'workflow',
                            'y_inference.csv')
    
    with open(x_inference_path, 'wb') as f:
        np.save(f, x_inference)

    response = requests.post(
        url=scanflow_uri,
        data = json.dumps(content)
    )    
    print(response)

    predictions = pd.read_csv(y_inference_path)
    
    return predictions['predictions'].values

def classify(image):

    image = image[:, :, 0]
    image = image[np.newaxis,...]
    
    print(image.shape)
    predictions = inference(image)
    prediction = predictions[0]

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
    ]
)

iface.launch(inline=False, debug=True) # Change inline=False in script.py