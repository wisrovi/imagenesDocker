# Dockerfile for serving Sphinx documentation
FROM nginx:alpine

# Copy the built documentation to nginx html directory
COPY _build/html /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]