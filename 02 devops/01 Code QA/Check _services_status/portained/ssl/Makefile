start:
	cd openssl && docker-compose up -d 
	
volumes_from:
	docker run -d \
  --name portainer \
  --restart always \
  --network portainer \
  -p 19443:9443 \
  -v /var/run/docker.sock:/var/run/docker.sock:Z \
  -v portainer:/data \
  -v $(pwd)/certs:/certs \
  --volumes-from worker \
  portainer/portainer-ce \
  --sslcert /certs/fullchain.pem \
  --sslkey /certs/privkey.pem