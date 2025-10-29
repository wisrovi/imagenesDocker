Examples
========

This section provides practical examples of using MinIO with Python, including data science workflows and DVC integration.

Python Setup
------------

Install required packages:

.. code-block:: bash

   pip install boto3 pandas scikit-learn dvc

Basic MinIO Operations with Python
----------------------------------

Connecting to MinIO
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import boto3
   from botocore.client import Config

   # MinIO configuration
   MINIO_ENDPOINT = 'http://localhost:30706'
   MINIO_ACCESS_KEY = 'DVC'
   MINIO_SECRET_KEY = 'uTAntEMTuVpcJucNjOJm'
   MINIO_BUCKET = 'datasets'

   # Create S3 client
   s3_client = boto3.client(
       's3',
       endpoint_url=MINIO_ENDPOINT,
       aws_access_key_id=MINIO_ACCESS_KEY,
       aws_secret_access_key=MINIO_SECRET_KEY,
       config=Config(signature_version='s3v4'),
       region_name='us-east-1'  # MinIO requires a region
   )

Creating a Bucket
~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       s3_client.create_bucket(Bucket=MINIO_BUCKET)
       print(f"Bucket '{MINIO_BUCKET}' created successfully")
   except s3_client.exceptions.BucketAlreadyExists:
       print(f"Bucket '{MINIO_BUCKET}' already exists")
   except Exception as e:
       print(f"Error creating bucket: {e}")

Uploading Files
~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Single File

      .. code-block:: python

         import os

         def upload_file(file_path, bucket_name, object_name=None):
             if object_name is None:
                 object_name = os.path.basename(file_path)

             try:
                 s3_client.upload_file(file_path, bucket_name, object_name)
                 print(f"File '{file_path}' uploaded as '{object_name}'")
             except Exception as e:
                 print(f"Error uploading file: {e}")

         # Usage
         upload_file('data.csv', MINIO_BUCKET)

   .. tab:: In-Memory Data

      .. code-block:: python

         import io

         def upload_dataframe(df, bucket_name, object_name):
             csv_buffer = io.StringIO()
             df.to_csv(csv_buffer, index=False)
             csv_buffer.seek(0)

             try:
                 s3_client.put_object(
                     Bucket=bucket_name,
                     Key=object_name,
                     Body=csv_buffer.getvalue()
                 )
                 print(f"DataFrame uploaded as '{object_name}'")
             except Exception as e:
                 print(f"Error uploading DataFrame: {e}")

         # Usage
         import pandas as pd
         df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
         upload_dataframe(df, MINIO_BUCKET, 'sample_data.csv')

Downloading Files
~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Download to File

      .. code-block:: python

         def download_file(bucket_name, object_name, file_path):
             try:
                 s3_client.download_file(bucket_name, object_name, file_path)
                 print(f"File '{object_name}' downloaded to '{file_path}'")
             except Exception as e:
                 print(f"Error downloading file: {e}")

         # Usage
         download_file(MINIO_BUCKET, 'data.csv', 'downloaded_data.csv')

   .. tab:: Load into DataFrame

      .. code-block:: python

         def load_csv_from_minio(bucket_name, object_name):
             try:
                 obj = s3_client.get_object(Bucket=bucket_name, Key=object_name)
                 df = pd.read_csv(io.BytesIO(obj['Body'].read()))
                 print(f"Loaded {len(df)} rows from '{object_name}'")
                 return df
             except Exception as e:
                 print(f"Error loading CSV: {e}")
                 return None

         # Usage
         df = load_csv_from_minio(MINIO_BUCKET, 'sample_data.csv')
         print(df.head())

Listing Objects
~~~~~~~~~~~~~~~

.. code-block:: python

   def list_objects(bucket_name, prefix=''):
       try:
           response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
           if 'Contents' in response:
               for obj in response['Contents']:
                   print(f"Object: {obj['Key']}, Size: {obj['Size']} bytes, Last Modified: {obj['LastModified']}")
           else:
               print("No objects found")
       except Exception as e:
           print(f"Error listing objects: {e}")

   # Usage
   list_objects(MINIO_BUCKET)

Machine Learning Workflow Example
---------------------------------

Complete ML pipeline with MinIO and DVC:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score
   import joblib
   import dvc.api

   def ml_pipeline():
       # Load data from MinIO
       print("Loading data from MinIO...")
       df = load_csv_from_minio(MINIO_BUCKET, 'raw_data.csv')

       # Preprocess data
       print("Preprocessing data...")
       X = df.drop('target', axis=1)
       y = df['target']
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       # Train model
       print("Training model...")
       model = RandomForestClassifier(n_estimators=100, random_state=42)
       model.fit(X_train, y_train)

       # Evaluate
       y_pred = model.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)
       print(f"Model accuracy: {accuracy:.4f}")

       # Save model locally
       joblib.dump(model, 'model.pkl')

       # Upload model to MinIO
       print("Uploading model to MinIO...")
       upload_file('model.pkl', MINIO_BUCKET, 'models/latest_model.pkl')

       # Track with DVC
       print("Tracking with DVC...")
       os.system('dvc add model.pkl')
       os.system('dvc push')

       print("Pipeline completed successfully!")

   if __name__ == "__main__":
       ml_pipeline()

DVC Integration Example
-----------------------

Using DVC parameters and metrics:

.. code-block:: python

   import yaml
   import dvc.api

   def dvc_ml_pipeline():
       # Load parameters from DVC
       params = dvc.api.params_show()
       n_estimators = params.get('n_estimators', 100)
       test_size = params.get('test_size', 0.2)

       # Load data path from DVC
       data_path = dvc.api.get_url('data', 'raw_data.csv')

       # Load data
       df = pd.read_csv(data_path)

       # Split data
       X = df.drop('target', axis=1)
       y = df['target']
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=test_size, random_state=42
       )

       # Train model
       model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
       model.fit(X_train, y_train)

       # Calculate metrics
       y_pred = model.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)

       # Save metrics to DVC
       metrics = {'accuracy': accuracy}
       with open('metrics.json', 'w') as f:
           json.dump(metrics, f)

       # Save model
       joblib.dump(model, 'model.pkl')

       print(f"Model trained with accuracy: {accuracy:.4f}")

   if __name__ == "__main__":
       dvc_ml_pipeline()

Batch Processing Example
------------------------

Processing multiple files in parallel:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   import glob

   def process_file(file_path):
       # Load and process individual file
       df = pd.read_csv(file_path)
       # ... processing logic ...
       processed_df = df  # placeholder

       # Upload processed file
       filename = os.path.basename(file_path)
       upload_dataframe(processed_df, MINIO_BUCKET, f'processed/{filename}')

       return f"Processed {filename}"

   def batch_process():
       # Find all CSV files
       csv_files = glob.glob('data/*.csv')

       # Process in parallel
       with ThreadPoolExecutor(max_workers=4) as executor:
           results = list(executor.map(process_file, csv_files))

       print("Batch processing completed:")
       for result in results:
           print(result)

   if __name__ == "__main__":
       batch_process()

Error Handling and Best Practices
---------------------------------

Robust error handling:

.. code-block:: python

   import logging
   from botocore.exceptions import ClientError

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   def safe_upload_file(file_path, bucket_name, object_name=None, max_retries=3):
       if object_name is None:
           object_name = os.path.basename(file_path)

       for attempt in range(max_retries):
           try:
               s3_client.upload_file(file_path, bucket_name, object_name)
               logger.info(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
               return True
           except ClientError as e:
               logger.warning(f"Attempt {attempt + 1} failed: {e}")
               if attempt == max_retries - 1:
                   logger.error(f"Failed to upload {file_path} after {max_retries} attempts")
                   return False
           except Exception as e:
               logger.error(f"Unexpected error uploading {file_path}: {e}")
               return False

       return False

   # Usage
   success = safe_upload_file('important_data.csv', MINIO_BUCKET)
   if not success:
       # Handle failure
       pass

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

Use environment variables or config files:

.. code-block:: python

   import os
   from dotenv import load_dotenv

   load_dotenv()

   class MinIOConfig:
       def __init__(self):
           self.endpoint = os.getenv('MINIO_ENDPOINT', 'http://localhost:30706')
           self.access_key = os.getenv('MINIO_ACCESS_KEY', 'DVC')
           self.secret_key = os.getenv('MINIO_SECRET_KEY', 'uTAntEMTuVpcJucNjOJm')
           self.bucket = os.getenv('MINIO_BUCKET', 'datasets')

       def get_client(self):
           return boto3.client(
               's3',
               endpoint_url=self.endpoint,
               aws_access_key_id=self.access_key,
               aws_secret_access_key=self.secret_key,
               config=Config(signature_version='s3v4'),
               region_name='us-east-1'
           )

   # Usage
   config = MinIOConfig()
   s3_client = config.get_client()