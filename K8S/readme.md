# Inference Server for YOLO Model

This project provides a FastAPI-based inference server for a YOLO model, allowing for video-based human detection.

## Project Structure

- `inference.py`: Contains the core API logic for loading the model, handling requests, and performing inference.
- `requirements.txt`: Lists the Python dependencies for the project.
- `Dockerfile`: Instructions for building the Docker image.
- `deployment.yaml`: Kubernetes deployment configuration for deploying the server on GKE.
- `Makefile`: Shortcuts for building, pushing the image, and deploying to GKE.
- `docker-compose.yaml`: Configuration for running the API locally using Docker Compose.

## Running the API Locally with Docker Compose

1. Install Docker and Docker Compose.
2. Set environment variables in `docker-compose.yaml` (replace placeholders with your model name and tag).
3. Run `docker-compose up` to start the API.
4. Access the API at `http://localhost:8080/infer`.

## Deployment to GKE

1. Follow instructions in `Makefile` to build, push the image, and deploy to GKE.
2. Set environment variables in `deployment.yaml` for the GKE environment.

## Usage

- Use `curl` or other HTTP clients to send POST requests to `/infer` with the video URI as a query parameter.
- The API will return a JSON response with the threshold and the confidence score for the "human_detected" class.