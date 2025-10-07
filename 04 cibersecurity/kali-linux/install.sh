# kali linux
sudo docker run -d --name kasmweb_kali --shm-size=512m -p 6901:6901 -e VNC_PW=password kasmweb/kali:develop
