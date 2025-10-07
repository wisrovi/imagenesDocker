git clone https://github.com/wazuh/wazuh-docker.git -b v4.11.2
cd wazuh-docker/single-node
docker-compose -f generate-indexer-certs.yml run --rm generator
docker-compose up -d --build

# The default username and password for the Wazuh dashboard are admin and SecretPassword. For additional security, you can change the default password for the Wazuh indexer admin user.



# dentro del indexer:
# yum install -y which
# export JAVA_HOME="/usr/share/wazuh-indexer/jdk"
# Opcional pero recomendado: añade también la ruta de Java al PATH del shell actual
# export PATH="$JAVA_HOME/bin:$PATH"
# bash /usr/share/wazuh-indexer/plugins/opensearch-security/tools/securityadmin.sh -cd /usr/share/wazuh-indexer/opensearch-security/ -icl -nhnv -cacert /usr/share/wazuh-indexer/certs/root-ca.pem  -cert /usr/share/wazuh-indexer/certs/admin.pem -key /usr/share/wazuh-indexer/certs/admin-key.pem -h wazuh.indexer -p 9200
