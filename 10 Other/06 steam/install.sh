# steam
sudo docker run -d --name kasmweb_steam --shm-size=512m -p 6901:6901 -e VNC_PW=password kasmweb/tor-browser:1.9.0
