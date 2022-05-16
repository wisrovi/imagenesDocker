sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt-get install curl -y
sudo apt autoremove -y

sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

curl -sSL https://get.docker.com | sh
sudo usermod -a -G docker $USER
sudo docker version

sudo docker run -d -p 8000:8000 -p 9000:9000 --name=portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce

curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo apt install docker-compose -y

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker


sudo docker run --gpus all --name=NVIDIA nvidia/cuda:9.0-base nvidia-smi

sudo docker pull tensorflow/tensorflow:latest-gpu
sudo docker pull tensorflow/tensorflow:latest-gpu-jupyter

sudo docker run -u $(id -u):$(id -g) --gpus all -d --name tensorflow -v $(pwd):/tf -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu-jupyter

sudo docker pull tensorflow/tensorflow:latest-gpu
sudo docker pull tensorflow/tensorflow:latest-gpu-jupyter

sudo reboot
