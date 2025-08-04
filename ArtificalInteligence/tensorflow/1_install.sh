# Install notepad y visual code
sudo apt install snap
sudo snap install notepad-plus-plus
sudo snap install --classic code

# install chrome
sudo apt install wget
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb

# install anydesk
sudo su
sudo wget -qO - https://keys.anydesk.com/repos/DEB-GPG-KEY | apt-key add -
echo "deb http://deb.anydesk.com/ all main" > /etc/apt/sources.list.d/anydesk-stable.list
apt update
apt install anydesk -y

# Install curl
sudo apt-get install -y curl

# instalando docker y python
sudo apt-get install -y docker docker-compose
sudo usermod -a -G docker $USER

sudo apt-get install -y python3-pip python3-dev
pip install tqdm
pip install selenium

# instalando portainer
sudo docker run -d -p 8000:8000 -p 9000:9000 --name=portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce

sudo docker pull tensorflow/tensorflow:latest-gpu
sudo docker pull tensorflow/tensorflow:latest-gpu-jupyter

# instalando drivers NVIDIA
# https://www.datamachinist.com/deep-learning/install-tensorflow-2-0-using-docker-with-gpu-support-on-ubuntu-18-04/
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot

# https://la.nvidia.com/Download/driverResults.aspx/193108/la
