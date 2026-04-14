# Informe: Despliegue de Kafka en Kubernetes con Strimzi

## Fecha
14 de Abril de 2026

## Visión General

Se ha desplegado un clúster de Kafka en el cluster Kubernetes existente utilizando el operador Strimzi.

---

## Componentes Desplegados

### 1. Namespace
- **Nombre**: `kafka`
- **Archivo**: `kafka_kube/00-namespace.yaml`

### 2. Operador Strimzi
- **Versión**: 0.51.0
- **Namespace**: kafka
- **CRDs instalados**:
  - `kafka.kafka.strimzi.io`
  - `kafkanodepool.kafka.strimzi.io`
  - `kafkatopics.kafka.strimzi.io`
  - `kafkausers.kafka.strimzi.io`
  - `kafkaconnects.kafka.strimzi.io`
  - Entre otros

### 3. Credenciales SMB
- **Secret**: `smb-credentials` en namespace `kafka`
- **Usuario**: `k8s_admin`
- **Ubicación**: `//192.168.20.99/kubernetes/kafka`
- **Archivo**: `kafka_kube/01-secret.yaml`

### 4. Kafka Cluster (Strimzi)
- **Nombre**: `kafka-cluster`
- **Versión Kafka**: 4.2.0
- **Brokers**: 3 (KafkaNodePool: broker-node-pool)
- **Controllers**: 3 (KafkaNodePool: controller-node-pool)
- **Archivo**: `kafka_kube/04-kafka.yaml`

### 5. KafkaNodePools
- **broker-node-pool**: 3 réplicas, rol broker
- **controller-node-pool**: 3 réplicas, rol controller
- **Archivo**: `kafka_kube/05-kafka-node-pool.yaml` y `kafka_kube/06-controller-node-pool.yaml`

---

## Almacenamiento

### Opción 1: Longhorn (Configuración Actual)
El cluster está configurado para usar **Longhorn** como storageClass por defecto.

```yaml
# Configuración de almacenamiento en KafkaNodePool
storage:
  type: jbod
  volumes:
    - id: 0
      type: persistent-claim
      size: 10Gi
    - id: 1
      type: persistent-claim
      size: 10Gi
    - id: 2
      type: persistent-claim
      size: 10Gi
```

- **Total**: 30Gi (3 brokers × 3 volúmenes × 10Gi)
- **PVCs creados automáticamente**: 9
- **StorageClass**: longhorn (por defecto)

### Opción 2: SMB (No disponible)
Se intentó configurar almacenamiento SMB (NAS) pero el driver CSI de SMB no está funcionando correctamente.

**Intentos realizados**:
1. StorageClass dinámico con SMB CSI driver - Falló (mount error 32)
2. PVs estáticos con SMB CSI driver - Funciona (PVCs Bound)

**Archivos SMB creados**:
- `kafka_kube/02-pv-static.yaml` - 9 PVs estáticos
- `kafka_kube/03-pvc-static.yaml` - 9 PVCs estáticos
- `kafka_kube/02-storageclass.yaml` - StorageClass SMB

**Problema**: El driver CSI de SMB (`smb.csi.k8s.io`) se instala correctamente pero los nodos no pueden montar los volúmenes. Posibles causas:
- Paquete `cifs-utils` no instalado en los nodos worker
- Permisos de red/accesso a la NAS
- Configuración de SMB en la NAS

---

## Recursos de Kafka

### Recursos por Broker/Controller
| Componente | CPU Request | CPU Limit | Memory Request | Memory Limit |
|------------|-------------|-----------|----------------|---------------|
| Broker     | 1           | 2         | 2Gi            | 4Gi           |
| Controller| 0.5         | 1         | 1Gi            | 2Gi           |

---

## Estado Final del Cluster

```
NAME                                             READY   STATUS
kafka-cluster-broker-node-pool-0                 1/1     Running
kafka-cluster-broker-node-pool-1                 1/1     Running
kafka-cluster-broker-node-pool-2                 1/1     Running
kafka-cluster-controller-node-pool-3             1/1     Running
kafka-cluster-controller-node-pool-4             1/1     Running
kafka-cluster-controller-node-pool-5             1/1     Running
kafka-cluster-entity-operator-...                  2/2     Running
strimzi-cluster-operator-...                      1/1     Running
```

### Estado de Kafka:
```bash
$ kubectl get kafka -n kafka
NAME            READY   WARNINGS   KAFKA VERSION   METADATA VERSION
kafka-cluster   True    True       4.2.0           4.2-IV1
```

---

## Servicios de Kafka

| Servicio | Endpoint | Puerto |
|----------|----------|--------|
| kafka-cluster-kafka-bootstrap | kafka-cluster-kafka-bootstrap.kafka.svc | 9092 |
| kafka-cluster-kafka-brokers | kafka-cluster-kafka-brokers.kafka.svc | 9090, 9091, 8443, 9092 |

---

## Estructura de Archivos

```
kafka_kube/
├── 00-namespace.yaml           # Namespace kafka
├── 01-secret.yaml              # Credenciales SMB
├── 02-pv.yaml                  # PVs originales (sin usar)
├── 02-pv-static.yaml          # PVs estáticos SMB (9 PVs)
├── 02-storageclass.yaml       # StorageClass SMB
├── 03-pvc.yaml                # PVCs originales (sin usar)
├── 03-pvc-static.yaml         # PVCs estáticos SMB (9 PVCs)
├── 04-kafka.yaml               # Kafka CR (Strimzi)
├── 05-kafka-node-pool.yaml    # KafkaNodePool broker
├── 06-controller-node-pool.yaml # KafkaNodePool controller
├── deploy.sh                   # Script de despliegue
├── test-pv.yaml               # PV de prueba
├── test-pvc.yaml              # PVC de prueba
└── INFORME.md                 # Este archivo
```

---

## Cómo Conectar a Kafka

### Desde dentro del cluster:
```bash
# Bootstrap servers
kafka-cluster-kafka-bootstrap.kafka.svc:9092
```

### Desde fuera del cluster (port-forward):
```bash
kubectl port-forward -n kafka svc/kafka-cluster-kafka-bootstrap 9092:9092
```

### Productor ejemplo:
```bash
kubectl run kafka-producer -n kafka --image=quay.io/strimzi/kafka:0.51.0-kafka-4.2.0 --rm -i --restart=Never -- bin/kafka-console-producer.sh --bootstrap-server kafka-cluster-kafka-bootstrap.kafka.svc:9092 --topic test
```

### Consumidor ejemplo:
```bash
kubectl run kafka-consumer -n kafka --image=quay.io/strimzi/kafka:0.51.0-kafka-4.2.0 --rm -i --restart=Never -- bin/kafka-console-consumer.sh --bootstrap-server kafka-cluster-kafka-bootstrap.kafka.svc:9092 --topic test --from-beginning
```

---

## Comandos de Gestión

### Ver estado del cluster:
```bash
kubectl get kafka -n kafka
kubectl get pods -n kafka
kubectl get pvc -n kafka
```

### Escalar brokers:
```bash
kubectl patch kafkanodepool broker-node-pool -n kafka -p '{"spec":{"replicas":5}}' --type=merge
```

### Eliminar Kafka:
```bash
kubectl delete -f kafka_kube/04-kafka.yaml
kubectl delete -f kafka_kube/05-kafka-node-pool.yaml
kubectl delete -f kafka_kube/06-controller-node-pool.yaml
```

### Eliminar todo:
```bash
kubectl delete -f kafka_kube/
```

---

## Notas

1. **Almacenamiento**: Actualmente Kafka usa Longhorn (30Gi total). Los datos se persisten en los volúmenes de Longhorn del cluster.

2. **Versión de Strimzi**: La versión instalada (0.51.0) usa el nuevo modelo de KafkaNodePool en lugar del modelo antiguo de Kafka CR. Esto requiere crear separados KafkaNodePools para brokers y controllers.

3. **SMB**: El driver CSI de SMB se instaló pero no funciona correctamente. Los PVCs dinámicos fallan con "mount failed: exit status 32". Los PVCs estáticos sí funcionan pero Strimzi no los usa automáticamente.

4. **Zookeeper**: No se usa ZooKeeper externo. Kafka 4.2+ usa KRaft (Kafka Raft) mode para el consenso.

5. **Seguridad**: Los listeners están configurados sin TLS para desarrollo. Para producción, configurar TLS/SASL.

---

## Próximos Pasos (Opcional)

1. **Configurar TLS**: Habilitar TLS en los listeners para producción
2. **Autenticación SASL**: Añadir autenticación de usuarios
3. **Conectar a NAS**: Investigar el problema de montaje SMB
4. **Monitoring**: Añadir Prometheus/Grafana para métricas de Kafka

---

## Referencias

- [Strimzi Documentation](https://strimzi.io/documentation/)
- [CSI Driver SMB](https://github.com/kubernetes-csi/csi-driver-smb)
- [Kafka KRaft](https://docs.confluent.io/platform/current/controller-boundary.html)
