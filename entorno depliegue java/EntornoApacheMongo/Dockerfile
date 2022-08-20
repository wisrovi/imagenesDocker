#FROM tomcat:8.0
FORM openjdk:8u292-jdk-slim-buster


#https://github.com/docker-library/tomcat

ADD prepago.war /usr/local/tomcat/webapps/

EXPOSE 8080
CMD ["catalina.sh", "run"]
