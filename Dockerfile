# Use an official Python runtime as a parent image
FROM python:3.10-bookworm

WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir .

#Expose uvicorn/fastapi
EXPOSE 8000

# Run entrypoint script when the container launches
CMD pl_aois >> /usr/src/app/aois_log.out 2>&1 
