# Configurando la NVIDA con el docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
sudo docker run --gpus all --name=NVIDIA nvidia/cuda:9.0-base nvidia-smi

# VEr si el tensorflow es capaz de ver la GPU
sudo docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu    python -c "import tensorflow as tf; print(tf.version); print(tf.test.is_gpu_available()); print(tf.test.is_built_with_cuda())"

# Instalando jupyter con soporte de la GPU
cd /Documents
sudo docker run -u $(id -u):$(id -g) --gpus all -d --name tensorflow -v $(pwd)/notebook:/tf -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu-jupyter
sudo chmod +777 -R notebook