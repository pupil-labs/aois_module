# Use an official Python runtime as a parent image
FROM python:3.10-bookworm

WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y \
	libgl1-mesa-dev \
	&& pip install --no-cache-dir .

#Expose uvicorn/fastapi
EXPOSE 8002

# Run entrypoint script when the container launches
CMD ["uvicorn", "pupil_labs.aois_module._defineAOIs:app", "--host", "0.0.0.0", "--port", "8002"]

