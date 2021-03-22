import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import math
import os
import plotly.graph_objects as go

from datetime import datetime

def plot_predictions(E_full, E_test, n_tests = 200, name_corr='anomaly'):
    n_train = n_tests
    E_train = E_full[E_full['Anomaly'].isna()].copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=E_full.tail(n_tests+n_train).index, y=E_full['Threshold_high'].tail(n_tests+n_train), 
                             name="Threshold",
                             line_color='dimgray'))
    fig.add_trace(go.Scatter(x=E_full.tail(n_tests+n_train).index, y=E_full['Threshold_low'].tail(n_tests+n_train), 
                             name="Threshold",
                             line_color='dimgray'))
    fig.add_trace(go.Scatter(x=E_train.tail(n_train).index, 
                             y=E_train['Loss_mae'].tail(n_train), name="Loss Training",
                             line_color='deepskyblue'))

    if all(E_test['Anomaly']):
        fig.add_trace(go.Scatter(x=E_test.index, y=E_test['Loss_mae'], name="Loss Testing",
                                 line_color='red'))
    elif not any(E_test['Anomaly']):
        fig.add_trace(go.Scatter(x=E_test.index, y=E_test['Loss_mae'], name="Loss Testing",
                                 line_color='green'))
    else:
        fig.add_trace(go.Scatter(x=E_test.index, y=E_test['Loss_mae'], name="Loss Testing",
                                 line_color='blue'))

    fig.update_layout(title_text=f'Checker prediction on test data, {name_corr}',
                      xaxis_rangeslider_visible=True)
    
    # fig.write_image(f"checker_anomaly_{name}.png")

    fig.show()
    
def show_numbers(x, y, n_display=9, title='MNIST'):
    n_cells = math.ceil(math.sqrt(n_display))
    fig = plt.figure(figsize=(5,5))

    for i in range(n_display):
        plt.subplot(n_cells, n_cells, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap='gray')
        # plt.imshow(x[i], cmap=plt.cm.binary)
        plt.xlabel(y[i])

    fig.suptitle(title)    
#     fig.savefig(f"{title}.png")
    plt.show()
    
def load_mnist_c(corruption='scale'):

    img_rows, img_cols = 28, 28
    x_train = np.load(f'mnist_c/{corruption}/train_images.npy')
    y_train = np.load(f'mnist_c/{corruption}/train_labels.npy')

    x_test = np.load(f'mnist_c/{corruption}/test_images.npy')
    y_test = np.load(f'mnist_c/{corruption}/test_labels.npy')

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)

    return (x_train, y_train), (x_test, y_test) 

def save_images(x):
    base_path = os.path.dirname(os.getcwd())
    images = os.path.join(base_path, "examples/demo_mnist/data-science/workflow/mnist_sample/test_images.npy")

    with open(images, 'wb') as f:
        np.save(f, x)
        
def sample_train_mnist():
    mnist_dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

    np.random.seed(42)

    sample_training = 1000
    x_idx = np.random.choice(range(len(x_train)), 
                                  size=sample_training, replace=False)
    x_train_sample, y_train_sample = x_train[x_idx], y_train[x_idx] 

    base_path = os.path.dirname(os.getcwd())
    x_train_path = os.path.join(base_path,
                                "examples/demo_mnist/data-science/workflow/mnist_sample/train_images.npy")
    with open(x_train_path, 'wb') as f:
        np.save(f, x_train_sample)

    y_train_path = os.path.join(base_path,
                                "examples/demo_mnist/data-science/workflow/mnist_sample/train_labels.npy")
    with open(y_train_path, 'wb') as f:
        np.save(f, y_train_sample)
        
    return x_train_sample, y_train_sample
        
def sample_test_mnist():
    mnist_dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
    np.random.seed(42)
    sample_testing = 1000
    x_idx = np.random.choice(range(len(x_test)), 
                                  size=sample_testing, replace=False)
    x_test_sample, y_test_sample = x_test[x_idx], y_test[x_idx] 

    base_path = os.path.dirname(os.getcwd())
    x_test_path = os.path.join(base_path,
                                "examples/demo_mnist/data-science/workflow/mnist_sample/test_images.npy")
    with open(x_test_path, 'wb') as f:
        np.save(f, x_test_sample)

    y_test_path = os.path.join(base_path,
                                "examples/demo_mnist/data-science/workflow/mnist_sample/test_labels.npy")
    with open(y_test_path, 'wb') as f:
        np.save(f, y_test_sample)
        
    return x_test_sample, y_test_sample


def sample_test_mnist_c(corruption='stripe'):

    (x_train, y_train), (x_test, y_test) = load_mnist_c(corruption)
    np.random.seed(42)
    sample_testing = 1000
    x_idx = np.random.choice(range(len(x_test)), 
                                  size=sample_testing, replace=False)
    x_test_sample, y_test_sample = x_test[x_idx], y_test[x_idx] 

    base_path = os.path.dirname(os.getcwd())
    x_test_path = os.path.join(base_path,
                                "examples/demo_mnist/data-science/workflow/mnist_sample/test_images.npy")
    with open(x_test_path, 'wb') as f:
        np.save(f, x_test_sample)

    y_test_path = os.path.join(base_path,
                                "examples/demo_mnist/data-science/workflow/mnist_sample/test_labels.npy")
    with open(y_test_path, 'wb') as f:
        np.save(f, y_test_sample)
        
    return x_test_sample, y_test_sample