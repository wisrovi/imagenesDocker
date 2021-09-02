https://www.datamachinist.com/deep-learning/install-tensorflow-2-0-using-docker-with-gpu-support-on-ubuntu-18-04/

sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot

curl -sSL https://get.docker.com | sh
sudo usermod -a -G docker $USER
docker version
sudo reboot

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

docker run --gpus all --rm nvidia/cuda nvidia-smi

docker pull tensorflow/tensorflow:latest-gpu-py3
docker pull tensorflow/tensorflow:latest-gpu-py3-jupyter

docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu-py3    python -c "import tensorflow as tf; print(tf.version); print(tf.test.is_gpu_available()); print(tf.test.is_built_with_cuda())"

docker run -u $(id -u):$(id -g) --gpus all -d --name tensorflow -v ~/docker_ws:/notebooks -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu-py3-jupyter

