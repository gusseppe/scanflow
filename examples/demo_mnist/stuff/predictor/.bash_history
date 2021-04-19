ls
ls
pwd
ls
pwd
cd ..
ls
mlflow deployments create -t torchserve -m
mlflow deployments create -t torchserve -m "models:/mnist_cnn:1" -n my_pytorch_model
mlflow deployments create -t torchserve -m "models:/mnist_cnn:1"
mlflow deployments create -t torchserve -m "models:/mnist_cnn:1" --name my_pytorch_model
pip install mlflow-torchserve
mlflow deployments create -t torchserve -m "models:/mnist_cnn:1" --name my_pytorch_model
pip install torch
mlflow deployments create -t torchserve -m "models:/mnist_cnn:1" --name my_pytorch_model
exit
