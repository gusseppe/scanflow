import os
import pandas as pd
import numpy as np
import imageio
import json
import glob
import gradio as gr
import requests

import sys
sys.path.insert(0,'../..')

model_name = sys.argv[1]
# model_version = int(sys.argv[2])


base_path = os.path.dirname(os.path.dirname(os.getcwd()))
app_dir = os.path.join(base_path, "examples/demo_mnist/")

scanflow_uri = 'http://localhost:8050/run/executor'
app_dir = '/home/guess/Desktop/scanflow/examples/demo_mnist/'
filename = 'x_inference_dashboard.npy'

content = {'name': 'inference-mnist-inference-batch',
          'parameters':{'model_name':model_name, 
                        'model_stage':"Production",
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
    
def classifier(folder):
    result = dict()
    images_numpy = list()
    images_path = list()
    
    images = []
    for im_path in glob.glob(f"{folder}/*.png"):
        im = imageio.imread(im_path)
        images_path.append(im_path)
        images_numpy.append(im)
     
    images_numpy = np.array(images_numpy)
    predictions = inference(images_numpy)
        
    result['image'] = images_path
    result['prediction'] = predictions
    
    df = pd.DataFrame(result)
    
    return df

textbox = gr.inputs.Textbox(label="Folder", default='images/mix')
iface = gr.Interface(classifier, textbox, "dataframe", server_port=7860, verbose=True)

iface.launch(inline=False, debug=True) # Change inline=False in script.py