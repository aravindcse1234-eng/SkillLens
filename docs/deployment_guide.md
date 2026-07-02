# SkillLens AI - Deployment Guide

## Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Docker & Docker Compose (optional)
- NVIDIA GPU + CUDA (optional, for acceleration)

## Local Deployment

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SkillLensAI.git
cd SkillLensAI

# Create virtual environment
python -m venv venv

# Activate it
# Windows: .\venv\Scripts\Activate
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"

# Configure environment
cp .env.example .env
# Edit .env with your database credentials
```

### 2. Database Setup

```bash
# Start PostgreSQL (Docker)
docker run --name skilllens-db \
    -e POSTGRES_DB=skilllens \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=postgres \
    -p 5432:5432 \
    -d postgres:15
```

### 3. Run Pipeline

```bash
# Quick training with sample data
python -m src.train_pipeline --mode quick

# Full pipeline (data collection + training)
python -m src.train_pipeline --mode full

# With MLflow tracking
python -m src.train_pipeline --mode quick --mlflow
```

### 4. Start Services

```bash
# Streamlit Dashboard (Terminal 1)
streamlit run app.py --server.port=8501

# FastAPI Backend (Terminal 2)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run Tests

```bash
pytest tests/ -v --cov=src
```

## Docker Deployment

### Build and Run

```bash
# Build and start all services
docker-compose up --build

# Access the dashboard
# http://localhost:8501

# Access the API
# http://localhost:8000/docs
```

### Production Configuration

For production, update the `.env` file:

```env
ENVIRONMENT=production
DEBUG=false
USE_GPU=false
DB_HOST=postgres
DB_PORT=5432
DB_NAME=skilllens
DB_USER=postgres
DB_PASSWORD=your_secure_password
```

## Cloud Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Set the main file path to `app.py`
5. Add secrets for database connection
6. Deploy

### AWS / GCP / Azure

1. Build Docker image:
   ```bash
   docker build -t skilllens-ai .
   ```

2. Push to container registry:
   ```bash
   docker tag skilllens-ai your-registry/skilllens-ai:latest
   docker push your-registry/skilllens-ai:latest
   ```

3. Deploy using:
   - **AWS**: ECS / EKS / Elastic Beanstalk
   - **GCP**: Cloud Run / GKE
   - **Azure**: Container Instances / AKS

## CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/ci_cd.yml`) that:

1. Lints code (black, flake8, isort)
2. Runs tests with PostgreSQL service
3. Trains models
4. Builds Docker image
5. Pushes to Docker Hub
6. Deploys to production

### Required Secrets

| Secret | Description |
|--------|-------------|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password or access token |

## Monitoring

- **Streamlit**: Built-in monitoring at `/_stcore/health`
- **FastAPI**: `/docs` for Swagger UI, `/redoc` for ReDoc
- **MLflow**: View experiment tracking at `http://localhost:5000`
- **Logs**: All logs stored in `logs/` directory

## Troubleshooting

### GPU Acceleration

If you have an NVIDIA GPU, set `USE_GPU=true` in `.env` and ensure CUDA is installed.

### Database Connection

If PostgreSQL is not available, the system falls back to sample data automatically.

### Model Performance Low

For best results, collect real job posting data using the data collection pipeline and train with a minimum of 10,000 samples.
