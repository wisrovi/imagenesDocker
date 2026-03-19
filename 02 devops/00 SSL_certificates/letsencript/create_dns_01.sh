docker run --rm -it -v ./nginx/certs/letsencrypt:/etc/letsencrypt certbot/certbot:v2.9.0 certonly \
  --manual \
  --preferred-challenges=dns \
  -d www.security.ecapturedtech.com \
  --email wrodriguez@ecapturedtech.com \
  --agree-tos \
  --no-eff-email