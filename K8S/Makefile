PROJECT_ID = # Replace with your GCP project ID
REGION = # Replace with your GCP region
IMAGE_NAME = inference-server

build:
  docker build -t $(IMAGE_NAME) .

push: build
  docker tag $(IMAGE_NAME) gcr.io/$(PROJECT_ID)/$(IMAGE_NAME)
  docker push gcr.io/$(PROJECT_ID)/$(IMAGE_NAME)

deploy: push
  kubectl apply -f deployment.yaml