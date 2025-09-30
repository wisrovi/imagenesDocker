# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install git and ssh client
RUN apt-get update && apt-get install -y git openssh-client

# Copy the entire project context to the working directory
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the application
RUN pip install .

# This command is a placeholder. The actual command will be provided at runtime.
CMD ["bash"]