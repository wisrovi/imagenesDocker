echo .
echo "direct test over cluster without security and without ingress"
curl http://10.43.233.14:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Responde solo con la palabra TEST"
}'

echo .
echo "Nginx (SSL) -> ModSecurity (WAF) -> ClusterIP -> Ollama Pods"
curl -k -X POST https://192.168.20.82:8300/ollama/api/generate   -H "Content-Type: application/json"   -d '{"model":"llama3.1:8b","prompt":"Hola","stream":false}'
