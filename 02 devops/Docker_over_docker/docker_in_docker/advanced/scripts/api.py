#!/usr/bin/env python3

"""
Simple REST API for Docker-in-Docker container management
Provides endpoints for managing containers, images, and volumes
"""

import os
import json
import subprocess
from flask import Flask, jsonify, request, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def run_docker_command(command):
    """Run a docker command and return the result"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            'success': result.returncode == 0,
            'output': result.stdout.strip(),
            'error': result.stderr.strip(),
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'error': 'Command timed out',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e),
            'returncode': -1
        }

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'dind-api',
        'version': '1.0.0'
    })

@app.route('/api/containers', methods=['GET'])
def list_containers():
    """List all containers"""
    result = run_docker_command('docker ps -a --format json')
    if result['success']:
        containers = []
        for line in result['output'].split('\n'):
            if line.strip():
                try:
                    containers.append(json.loads(line))
                except:
                    pass
        return jsonify({
            'containers': containers,
            'count': len(containers)
        })
    return jsonify({'error': result['error']}), 500

@app.route('/api/containers/<container_id>', methods=['GET'])
def get_container(container_id):
    """Get container details"""
    result = run_docker_command(f'docker inspect {container_id}')
    if result['success']:
        try:
            details = json.loads(result['output'])
            return jsonify(details[0] if details else {})
        except:
            return jsonify({'error': 'Invalid JSON response'}), 500
    return jsonify({'error': result['error']}), 404

@app.route('/api/containers/<container_id>/start', methods=['POST'])
def start_container(container_id):
    """Start a container"""
    result = run_docker_command(f'docker start {container_id}')
    if result['success']:
        return jsonify({'message': f'Container {container_id} started'})
    return jsonify({'error': result['error']}), 500

@app.route('/api/containers/<container_id>/stop', methods=['POST'])
def stop_container(container_id):
    """Stop a container"""
    result = run_docker_command(f'docker stop {container_id}')
    if result['success']:
        return jsonify({'message': f'Container {container_id} stopped'})
    return jsonify({'error': result['error']}), 500

@app.route('/api/containers/<container_id>/restart', methods=['POST'])
def restart_container(container_id):
    """Restart a container"""
    result = run_docker_command(f'docker restart {container_id}')
    if result['success']:
        return jsonify({'message': f'Container {container_id} restarted'})
    return jsonify({'error': result['error']}), 500

@app.route('/api/containers/<container_id>/logs', methods=['GET'])
def get_container_logs(container_id):
    """Get container logs"""
    lines = request.args.get('lines', default=100, type=int)
    result = run_docker_command(f'docker logs --tail {lines} {container_id}')
    if result['success']:
        return jsonify({
            'container_id': container_id,
            'logs': result['output']
        })
    return jsonify({'error': result['error']}), 500

@app.route('/api/images', methods=['GET'])
def list_images():
    """List all Docker images"""
    result = run_docker_command('docker images --format json')
    if result['success']:
        images = []
        for line in result['output'].split('\n'):
            if line.strip():
                try:
                    images.append(json.loads(line))
                except:
                    pass
        return jsonify({
            'images': images,
            'count': len(images)
        })
    return jsonify({'error': result['error']}), 500

@app.route('/api/volumes', methods=['GET'])
def list_volumes():
    """List all Docker volumes"""
    result = run_docker_command('docker volume ls --format json')
    if result['success']:
        volumes = []
        for line in result['output'].split('\n'):
            if line.strip():
                try:
                    volumes.append(json.loads(line))
                except:
                    pass
        return jsonify({
            'volumes': volumes,
            'count': len(volumes)
        })
    return jsonify({'error': result['error']}), 500

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get Docker system information"""
    result = run_docker_command('docker system info --format json')
    if result['success']:
        try:
            info = json.loads(result['output'])
            return jsonify(info)
        except:
            return jsonify({'error': 'Invalid JSON response'}), 500
    return jsonify({'error': result['error']}), 500

@app.route('/api/system/df', methods=['GET'])
def system_df():
    """Get Docker system disk usage"""
    result = run_docker_command('docker system df --format json')
    if result['success']:
        try:
            df = json.loads(result['output'])
            return jsonify(df)
        except:
            return jsonify({'error': 'Invalid JSON response'}), 500
    return jsonify({'error': result['error']}), 500

@app.route('/api/backup', methods=['POST'])
def create_backup():
    """Create a backup of volumes"""
    result = run_docker_command('/usr/local/bin/backup.sh')
    if result['success']:
        return jsonify({'message': 'Backup created successfully'})
    return jsonify({'error': result['error']}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('API_PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)