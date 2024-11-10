# Variables
IMAGE_NAME = segmentation-model
CONTAINER_NAME = segmentation-container
PORT = 8000

# Default target
.PHONY: all
all: clean build run

# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

# Run the container with default settings
.PHONY: run
run: stop
	docker run -d --name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		$(IMAGE_NAME)
	@echo "Container started. API available at http://localhost:$(PORT)/docs"

# Run with custom model and image (usage: make run-custom MODEL=model_name IMAGE=path/to/image USE_ONNX=True)
.PHONY: run-custom
run-custom: stop
	docker run -d --name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		-e MODEL_NAME="$(MODEL)" \
		-e IMAGE_PATH="$(IMAGE)" \
		-e USE_ONNX="$(USE_ONNX)" \
		$(IMAGE_NAME)
	@echo "Container started with custom settings. API available at http://localhost:$(PORT)/docs"

# Run with mounted volumes for models and images
.PHONY: run-mounted
run-mounted: stop
	docker run -d --name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/images:/app/images \
		$(IMAGE_NAME)
	@echo "Container started with mounted volumes. API available at http://localhost:$(PORT)/docs"

# Stop and remove the container
.PHONY: stop
stop:
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true

# Clean up: stop container and remove image
.PHONY: clean
clean: stop
	-docker rmi $(IMAGE_NAME) 2>/dev/null || true

# Show container logs
.PHONY: logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Enter container shell
.PHONY: shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Restart the container
.PHONY: restart
restart: stop run

# Show container status
.PHONY: status
status:
	@docker ps -a | grep $(CONTAINER_NAME) || echo "Container not found"