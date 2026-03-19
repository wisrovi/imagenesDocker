Examples
========

This section provides practical examples of using MinIO with various tools and programming languages.

Python Examples
---------------

Basic S3 Operations with boto3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import boto3
   from botocore.exceptions import ClientError

   class MinIOClient:
       def __init__(self, endpoint_url='http://localhost:30706',
                    access_key='DVC', secret_key='uTAntEMTuVpcJucNjOJm'):
           self.s3_client = boto3.client(
               's3',
               endpoint_url=endpoint_url,
               aws_access_key_id=access_key,
               aws_secret_access_key=secret_key,
               region_name='us-east-1'
           )

       def create_bucket(self, bucket_name):
           """Create a new bucket"""
           try:
               self.s3_client.create_bucket(Bucket=bucket_name)
               print(f"Bucket '{bucket_name}' created successfully")
           except ClientError as e:
               print(f"Error creating bucket: {e}")

       def upload_file(self, file_path, bucket_name, object_key):
           """Upload a file to a bucket"""
           try:
               self.s3_client.upload_file(file_path, bucket_name, object_key)
               print(f"File '{file_path}' uploaded as '{object_key}'")
           except ClientError as e:
               print(f"Error uploading file: {e}")

       def download_file(self, bucket_name, object_key, file_path):
           """Download a file from a bucket"""
           try:
               self.s3_client.download_file(bucket_name, object_key, file_path)
               print(f"File '{object_key}' downloaded to '{file_path}'")
           except ClientError as e:
               print(f"Error downloading file: {e}")

       def list_objects(self, bucket_name):
           """List all objects in a bucket"""
           try:
               response = self.s3_client.list_objects_v2(Bucket=bucket_name)
               if 'Contents' in response:
                   for obj in response['Contents']:
                       print(f"Object: {obj['Key']}, Size: {obj['Size']} bytes")
               else:
                   print("Bucket is empty")
           except ClientError as e:
               print(f"Error listing objects: {e}")

   # Usage example
   if __name__ == "__main__":
       minio = MinIOClient()
       minio.create_bucket('my-example-bucket')
       minio.upload_file('example.txt', 'my-example-bucket', 'uploaded_example.txt')
       minio.list_objects('my-example-bucket')

Machine Learning Data Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   from minio import Minio
   from io import BytesIO

   class MLDataManager:
       def __init__(self, endpoint='localhost:30706',
                    access_key='DVC', secret_key='uTAntEMTuVpcJucNjOJm',
                    secure=False):
           self.minio_client = Minio(
               endpoint,
               access_key=access_key,
               secret_key=secret_key,
               secure=secure
           )

       def upload_dataframe(self, df, bucket_name, object_name):
           """Upload a pandas DataFrame as CSV"""
           csv_buffer = BytesIO()
           df.to_csv(csv_buffer, index=False)
           csv_buffer.seek(0)

           self.minio_client.put_object(
               bucket_name,
               object_name,
               csv_buffer,
               length=csv_buffer.getbuffer().nbytes,
               content_type='text/csv'
           )
           print(f"DataFrame uploaded as {object_name}")

       def download_dataframe(self, bucket_name, object_name):
           """Download and return a pandas DataFrame"""
           response = self.minio_client.get_object(bucket_name, object_name)
           df = pd.read_csv(response)
           return df

       def upload_numpy_array(self, array, bucket_name, object_name):
           """Upload a NumPy array as NPY file"""
           npy_buffer = BytesIO()
           np.save(npy_buffer, array)
           npy_buffer.seek(0)

           self.minio_client.put_object(
               bucket_name,
               object_name,
               npy_buffer,
               length=npy_buffer.getbuffer().nbytes,
               content_type='application/octet-stream'
           )
           print(f"NumPy array uploaded as {object_name}")

       def download_numpy_array(self, bucket_name, object_name):
           """Download and return a NumPy array"""
           response = self.minio_client.get_object(bucket_name, object_name)
           array = np.load(BytesIO(response.read()))
           return array

   # Usage example
   if __name__ == "__main__":
       manager = MLDataManager()

       # Create sample data
       df = pd.DataFrame({
           'feature1': np.random.randn(100),
           'feature2': np.random.randn(100),
           'target': np.random.randint(0, 2, 100)
       })

       array = np.random.rand(10, 10)

       # Upload data
       manager.upload_dataframe(df, 'ml-bucket', 'dataset.csv')
       manager.upload_numpy_array(array, 'ml-bucket', 'model_weights.npy')

       # Download data
       downloaded_df = manager.download_dataframe('ml-bucket', 'dataset.csv')
       downloaded_array = manager.download_numpy_array('ml-bucket', 'model_weights.npy')

       print(f"Downloaded DataFrame shape: {downloaded_df.shape}")
       print(f"Downloaded array shape: {downloaded_array.shape}")

DVC Integration Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   import subprocess
   from pathlib import Path

   class DVCManager:
       def __init__(self, minio_endpoint='http://localhost:30706',
                    access_key='DVC', secret_key='uTAntEMTuVpcJucNjOJm'):
           self.minio_endpoint = minio_endpoint
           self.access_key = access_key
           self.secret_key = secret_key

       def init_dvc(self):
           """Initialize DVC in the current directory"""
           subprocess.run(['dvc', 'init'], check=True)
           print("DVC initialized")

       def add_remote(self, remote_name, bucket_name):
           """Add MinIO as a DVC remote"""
           subprocess.run([
               'dvc', 'remote', 'add', '-d', remote_name,
               f's3://{bucket_name}'
           ], check=True)

           subprocess.run([
               'dvc', 'remote', 'modify', remote_name,
               'endpointurl', self.minio_endpoint
           ], check=True)

           subprocess.run([
               'dvc', 'remote', 'modify', remote_name,
               'access_key_id', self.access_key
           ], check=True)

           subprocess.run([
               'dvc', 'remote', 'modify', remote_name,
               'secret_access_key', self.secret_key
           ], check=True)

           print(f"Remote '{remote_name}' configured")

       def track_file(self, file_path):
           """Track a file with DVC"""
           subprocess.run(['dvc', 'add', file_path], check=True)
           print(f"File '{file_path}' tracked")

       def push_data(self):
           """Push tracked data to remote"""
           subprocess.run(['dvc', 'push'], check=True)
           print("Data pushed to remote")

       def pull_data(self):
           """Pull data from remote"""
           subprocess.run(['dvc', 'pull'], check=True)
           print("Data pulled from remote")

   # Usage example
   if __name__ == "__main__":
       dvc_manager = DVCManager()

       # Initialize DVC project
       dvc_manager.init_dvc()

       # Configure remote
       dvc_manager.add_remote('myremote', 'dvc-bucket')

       # Track and push data
       dvc_manager.track_file('data/dataset.csv')
       dvc_manager.push_data()

       # In another environment, pull the data
       # dvc_manager.pull_data()

JavaScript/Node.js Examples
---------------------------

Using AWS SDK for JavaScript
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: javascript

   const AWS = require('aws-sdk');
   const fs = require('fs');

   // Configure AWS SDK for MinIO
   const s3 = new AWS.S3({
     endpoint: 'http://localhost:30706',
     accessKeyId: 'DVC',
     secretAccessKey: 'uTAntEMTuVpcJucNjOJm',
     s3ForcePathStyle: true,
     signatureVersion: 'v4'
   });

   async function createBucket(bucketName) {
     try {
       await s3.createBucket({ Bucket: bucketName }).promise();
       console.log(`Bucket '${bucketName}' created successfully`);
     } catch (error) {
       console.error('Error creating bucket:', error);
     }
   }

   async function uploadFile(filePath, bucketName, key) {
     try {
       const fileContent = fs.readFileSync(filePath);
       await s3.putObject({
         Bucket: bucketName,
         Key: key,
         Body: fileContent
       }).promise();
       console.log(`File '${filePath}' uploaded as '${key}'`);
     } catch (error) {
       console.error('Error uploading file:', error);
     }
   }

   async function listObjects(bucketName) {
     try {
       const response = await s3.listObjectsV2({ Bucket: bucketName }).promise();
       if (response.Contents) {
         response.Contents.forEach(obj => {
           console.log(`Object: ${obj.Key}, Size: ${obj.Size} bytes`);
         });
       } else {
         console.log('Bucket is empty');
       }
     } catch (error) {
       console.error('Error listing objects:', error);
     }
   }

   // Usage
   async function main() {
     await createBucket('my-js-bucket');
     await uploadFile('example.txt', 'my-js-bucket', 'uploaded-example.txt');
     await listObjects('my-js-bucket');
   }

   main();

Bash Scripting Examples
-----------------------

Automated Backup Script
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash

   # MinIO Backup Script
   MINIO_ALIAS="myminio"
   ENDPOINT="http://localhost:30706"
   ACCESS_KEY="DVC"
   SECRET_KEY="uTAntEMTuVpcJucNjOJm"
   BUCKET_NAME="backup-bucket"
   BACKUP_DIR="/path/to/backup"

   # Configure MinIO client
   mc alias set $MINIO_ALIAS $ENDPOINT $ACCESS_KEY $SECRET_KEY

   # Create bucket if it doesn't exist
   mc mb $MINIO_ALIAS/$BUCKET_NAME --ignore-existing

   # Create timestamp
   TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

   # Compress backup directory
   tar -czf ${BACKUP_DIR}_${TIMESTAMP}.tar.gz -C /path/to $BACKUP_DIR

   # Upload to MinIO
   mc cp ${BACKUP_DIR}_${TIMESTAMP}.tar.gz $MINIO_ALIAS/$BUCKET_NAME/

   # Clean up local backup file
   rm ${BACKUP_DIR}_${TIMESTAMP}.tar.gz

   echo "Backup completed: ${BACKUP_DIR}_${TIMESTAMP}.tar.gz"

Sync Directory Script
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash

   # Directory Sync Script for MinIO
   MINIO_ALIAS="myminio"
   ENDPOINT="http://localhost:30706"
   ACCESS_KEY="DVC"
   SECRET_KEY="uTAntEMTuVpcJucNjOJm"
   BUCKET_NAME="sync-bucket"
   LOCAL_DIR="/path/to/local/dir"

   # Configure MinIO client
   mc alias set $MINIO_ALIAS $ENDPOINT $ACCESS_KEY $SECRET_KEY

   # Create bucket if it doesn't exist
   mc mb $MINIO_ALIAS/$BUCKET_NAME --ignore-existing

   # Sync local directory to MinIO bucket
   mc mirror --overwrite $LOCAL_DIR $MINIO_ALIAS/$BUCKET_NAME/

   echo "Directory sync completed"

Docker Compose Examples
-----------------------

Multi-Service Setup
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   version: '3.8'

   services:
     minio:
       image: minio/minio:RELEASE.2025-02-28T09-55-16Z
       command: server /data --console-address ":9001"
       environment:
         - MINIO_ROOT_USER=DVC
         - MINIO_ROOT_PASSWORD=uTAntEMTuVpcJucNjOJm
       ports:
         - "30706:9000"
         - "30707:9001"
       volumes:
         - ./data:/data
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
         interval: 30s
         timeout: 20s
         retries: 3

     ml-app:
       image: my-ml-app:latest
       environment:
         - MINIO_ENDPOINT=http://minio:9000
         - MINIO_ACCESS_KEY=DVC
         - MINIO_SECRET_KEY=uTAntEMTuVpcJucNjOJm
       depends_on:
         minio:
           condition: service_healthy
       volumes:
         - ./app:/app

     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
       depends_on:
         - ml-app

CI/CD Integration
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # .github/workflows/ml-pipeline.yml
   name: ML Pipeline

   on: [push]

   jobs:
     train:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v2
       - name: Set up Python
         uses: actions/setup-python@v2
         with:
           python-version: '3.8'
       - name: Install dependencies
         run: |
           pip install -r requirements.txt
           pip install dvc
       - name: Start MinIO
         run: |
           docker run -d -p 30706:9000 -p 30707:9001 \
             -e MINIO_ROOT_USER=DVC \
             -e MINIO_ROOT_PASSWORD=uTAntEMTuVpcJucNjOJm \
             minio/minio:RELEASE.2025-02-28T09-55-16Z server /data
       - name: Configure DVC
         run: |
           dvc remote add -d myremote s3://ml-bucket
           dvc remote modify myremote endpointurl http://localhost:30706
           dvc remote modify myremote access_key_id DVC
           dvc remote modify myremote secret_access_key uTAntEMTuVpcJucNjOJm
       - name: Pull data
         run: dvc pull
       - name: Train model
         run: python train.py
       - name: Push results
         run: dvc push