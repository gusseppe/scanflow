import os
import pandas as pd
import numpy as np
import imageio
import json
import glob
import gradio as gr
from zipfile import ZipFile


import sys
sys.path.insert(0,'../..')
from scanflow import tools
from scanflow.deploy import Deploy

base_path = os.path.dirname(os.path.dirname(os.getcwd()))
app_dir = os.path.join(base_path, "examples/demo_mnist/")

content = {'app_dir': app_dir, 'name': 'inference-mnist'}

def inference(x_inference):
    filename = 'x_inference_dashboard.npy'
    x_inference_path = os.path.join(content['app_dir'], 'workflow',
                                filename)
    y_inference_path = os.path.join(content['app_dir'], 'workflow',
                            'y_inference.csv')
    with open(x_inference_path, 'wb') as f:
        np.save(f, x_inference)

    paths = tools.get_scanflow_paths(content['app_dir'])
    meta_dir = paths['meta_dir']

    workflows_metadata_name = f"{content['name']}.json"
    workflows_metadata_path = os.path.join(meta_dir, workflows_metadata_name)

    with open(workflows_metadata_path) as fread:
        setup_json = json.load(fread)
        
    setup_json['executors'][0]['parameters']['x_inference_path'] = filename
    
    deployer = Deploy(setup_json)
    deployer.run_workflows(verbose=True)
#     result = deployer.logs_run_workflow[0]['envs'][0]['result']
    
    predictions = pd.read_csv(y_inference_path)
    
    return predictions['predictions'].values
 
def trigger_mas():
    import requests

    url = 'http://localhost:8003/send/checker/anomaly'
    response = requests.get(
        url=url,
        headers={"accept": "application/json"})

    response_json = json.loads(response.text)
    print(response_json)
    
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
iface = gr.Interface(classifier, textbox, "dataframe")

iface.launch(inline=False, debug=True) # Change inline=False in script.py