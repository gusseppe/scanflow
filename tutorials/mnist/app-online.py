import gradio as gr
import numpy as np
import json
# sys.path.insert(1,'../..')

# from scanflow.deploy import Deploy

# base_path = os.path.dirname(os.path.dirname(os.getcwd()))
# app_dir = os.path.join(base_path, "examples/demo_mnist/")

def classify(image):
#     prediction = model.predict(image).tolist()[0]
    prediction = np.random.randint(10)
#     return {str(i): prediction[i] for i in range(10)}
    return  prediction

iface = gr.Interface(
    classify, 
    gr.inputs.Image(shape=(100, 100)), 
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