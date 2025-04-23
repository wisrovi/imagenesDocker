https://www.datamachinist.com/deep-learning/install-tensorflow-2-0-using-docker-with-gpu-support-on-ubuntu-18-04/

sudo apt-get install ubuntu-drivers-common 
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot

curl -sSL https://get.docker.com | sh
sudo usermod -a -G docker $USER
docker version
sudo reboot

VERSION=$(curl --silent https://api.github.com/repos/docker/compose/releases/latest | grep -Po '"tag_name": "\K.*\d')
DESTINATION=/usr/local/bin/docker-compose
sudo curl -L https://github.com/docker/compose/releases/download/${VERSION}/docker-compose-$(uname -s)-$(uname -m) -o $DESTINATION
sudo chmod 755 $DESTINATION
docker-compose --version

# instalar portainer
docker run -d -p 8000:8000 -p 9000:9000 --name=portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

docker pull nvidia/cuda:12.3.1-devel-ubuntu22.04

# una prueba basica
docker run --rm --gpus all --name=NVIDIA nvidia/cuda:12.3.1-devel-ubuntu22.04 nvidia-smi

# un contenedor con estado de salud que muestra cuando se han perdido las GPU
docker run -d --name NVIDIA --gpus all --health-cmd="nvidia-smi || exit 1" --health-interval=30s --health-retries=3 --health-timeout=5s nvidia/cuda:12.2.0-base-ubuntu22.04 bash -c "while true; do nvidia-smi || break; sleep 30; done; tail -f /dev/null"




docker pull tensorflow/tensorflow:latest-gpu
docker pull tensorflow/tensorflow:latest-gpu-jupyter

docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu    python -c "import tensorflow as tf; print(tf.version); print(tf.test.is_gpu_available()); print(tf.test.is_built_with_cuda())"

# instalar jupyter
docker run -u $(id -u):$(id -g) --gpus all -d --name tensorflow -v /home/rh/Documentos/docker/tensorflow:/tf -p 8888:8888 -p 6006:6006 --user root -e GRANT_SUDO=yes -e NB_GID=100 -e GEN_CERT=yes   tensorflow/tensorflow:latest-gpu-jupyter

# jupyter para kaggle
docker run -u $(id -u):$(id -g) --gpus all -d --name TFM -v /media/wisrovi/J/TFM/2022/dataset/archive/musicnet/kaggle:/kaggle -v /media/wisrovi/J/TFM/2022/dataset/archive/musicnet/tf:/tf  -p 8889:8888 -p 6009:6006 --user root -e GRANT_SUDO=yes -e NB_GID=100 -e GEN_CERT=yes  tensorflow/tensorflow:latest-gpu-jupyter
#adentro del contenedor:
apt-get install libsndfile1



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


sudo apt install snap
sudo snap install notepad-plus-plus




