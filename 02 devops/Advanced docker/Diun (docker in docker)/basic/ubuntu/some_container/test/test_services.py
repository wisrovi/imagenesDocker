import pytest
import requests

BASE_URL_HTTP = "http://192.168.1.84:50424"
BASE_URL_HTTPS = "https://192.168.1.84:50425"
BASE_URL_PORTAINER = "http://192.168.1.84:50421"
BASE_URL_TTYD = "http://192.168.1.84:50423"

@pytest.mark.parametrize("worker,path", [
    ("portainer1", "/portainer1/"),
    ("portainer2", "/portainer2/"),
    ("portainer3", "/portainer3/"),
    ("portainer4", "/portainer4/"),
    ("portainer5", "/portainer5/"),
    ("ttyd1", "/ttyd1/"),
    ("ttyd2", "/ttyd2/"),
    ("ttyd3", "/ttyd3/"),
    ("ttyd4", "/ttyd4/"),
    ("ttyd5", "/ttyd5/"),
    ("worker1_http", "/worker1/"),
    ("worker1_https", "/worker1/"),
    ("worker2_http", "/worker2/"),
    ("worker2_https", "/worker2/"),
    ("worker3_http", "/worker3/"),
    ("worker3_https", "/worker3/"),
    ("worker4_http", "/worker4/"),
    ("worker4_https", "/worker4/"),
    ("worker5_http", "/worker5/"),
    ("worker5_https", "/worker5/"),
])
def test_service_access(worker, path):
    if worker.startswith('portainer'):
        base_url = BASE_URL_PORTAINER
    elif worker.startswith('ttyd'):
        base_url = BASE_URL_TTYD
    elif 'https' in worker:
        base_url = BASE_URL_HTTPS
    else:
        base_url = BASE_URL_HTTP
    url = f"{base_url}{path}"
    try:
        response = requests.get(url, verify=False, timeout=10)
        assert response.status_code == 200, f"Failed for {worker}: {response.status_code}"
    except requests.RequestException as e:
        pytest.fail(f"Request failed for {worker}: {e}")