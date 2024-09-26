# LLM Inference Microservice

This project implements a microservice for sentiment analysis on product reviews using a Large Language Model (LLM). It includes advanced pre-processing. The project now supports deployment using Kubernetes and Helm.

## Features

- Custom inference pipeline for sentiment analysis
- FastAPI web service for easy integration
- Single review and batch processing endpoints
- Docker Compose setup for easy local deployment
- Kubernetes deployment support using Helm charts

## Prerequisites

- Docker and Docker Compose (for local development)
- Kubernetes cluster (for production deployment)
- Helm (for Kubernetes package management)

## Project Structure

```
.
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
├── helm
│   └── llm-service
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates
│           ├── _helpers.tpl
│           ├── deployment.yaml
│           ├── service.yaml
│           └── ingress.yaml
└── src
    ├── apis
    │   ├── api.py
    ├── pipeline
    │   ├── customIn_ference_pipeline.py
    ├── schemas
    │   └── reviews.py
    ├── services
    │   ├── analyze.py
    ├── __init__.py
    ├── main.py
```

## Quick Start (Local Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/rishibrainerhub/llm-microservice-kubernetes-helm.git
   cd llm-microservice-kubernetes-helm
   ```

2. Build and start the services:
   ```bash
   docker-compose up --build
   ```

3. The service will be available at `http://localhost:8000`

## Kubernetes Deployment

1. Ensure you have a Kubernetes cluster running and `kubectl` configured to communicate with your cluster.

2. Install Helm if you haven't already:
   ```bash
   curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
   ```

3. Package the Helm chart:
   ```bash
   helm package helm/llm-service
   ```

4. Install the Helm chart:
   ```bash
   helm install llm-service ./llm-service-0.1.0.tgz
   ```

5. Verify the deployment:
   ```bash
   kubectl get pods
   ```

6. Access the service:
   If you're using Minikube, you may need to use port-forwarding:
   ```bash
   kubectl port-forward service/llm-service 8000:80
   ```
   The service will then be available at `http://localhost:8000`

## API Endpoints

### 1. Analyze Single Review

- **URL**: `/analyze_review`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Your review text here"
  }
  ```
- **Response**: JSON object with analysis results

### 2. Analyze Batch of Reviews

- **URL**: `/analyze_batch`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "reviews": ["Review 1 text", "Review 2 text", ...]
  }
  ```
- **Response**: Array of JSON objects with analysis results

## Custom Inference Pipeline

The custom inference pipeline (`custom_inference_pipeline.py`) includes the following steps:

1. Text preprocessing (HTML removal, special character removal)
2. Sentiment analysis using a pre-trained LLM
3. Named entity extraction
4. Result postprocessing

## Helm Chart Configuration

The Helm chart for this project is located in the `helm/llm-service` directory. Key configuration options in `values.yaml` include:

- `replicaCount`: Number of replicas for the deployment
- `image.repository` and `image.tag`: Docker image details
- `ingress`: Configuration for Ingress resource
- `resources`: CPU and memory requests/limits

Modify these values as needed for your deployment.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.