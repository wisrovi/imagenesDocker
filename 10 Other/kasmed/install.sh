# kasmweb
sudo docker run -d --name kasmweb --shm-size=1024m -v /dev/snd:/dev/snd -v /dev/shm:/dev/shm  -p 6901:6901 -e VNC_PW=password kasmweb/desktop-deluxe:develop-rolling

#User : kasm_user
#Password: password
