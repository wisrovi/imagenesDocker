https://www.datamachinist.com/deep-learning/install-tensorflow-2-0-using-docker-with-gpu-support-on-ubuntu-18-04/

sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot

curl -sSL https://get.docker.com | sh
sudo usermod -a -G docker $USER
docker version
sudo reboot

docker run -d -p 8000:8000 -p 9000:9000 --name=portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce

curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose


distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

docker run --gpus all --name=NVIDIA nvidia/cuda:9.0-base nvidia-smi

docker pull tensorflow/tensorflow:latest-gpu
docker pull tensorflow/tensorflow:latest-gpu-jupyter

docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu    python -c "import tensorflow as tf; print(tf.version); print(tf.test.is_gpu_available()); print(tf.test.is_built_with_cuda())"

docker run -u $(id -u):$(id -g) --gpus all -d --name tensorflow -v /home/rh/Documentos/docker/tensorflow:/tf -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu-jupyter

# otras instalaciones
sudo apt install python3-pip -y
pip install tqdm
pip install selenium

sudo su

sudo apt install wget
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb

wget -qO - https://keys.anydesk.com/repos/DEB-GPG-KEY | apt-key add -
echo "deb http://deb.anydesk.com/ all main" > /etc/apt/sources.list.d/anydesk-stable.list
apt update
apt install anydesk -y






