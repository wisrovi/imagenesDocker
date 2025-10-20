FROM python:3.9-alpine AS builder

# Install make and Sphinx with themes and extensions
RUN apk add --no-cache make && pip install --no-cache-dir sphinx sphinx-rtd-theme sphinx-tabs

# Copy documentation source
COPY . /docs

# Set working directory
WORKDIR /docs

# Build the HTML documentation
RUN make html

# Use nginx to serve the built documentation
FROM nginx:alpine

# Copy built HTML from builder stage
COPY --from=builder /docs/_build/html /usr/share/nginx/html

# Expose port 80
EXPOSE 80