# Deployment

Deployment configurations for SkillLens.

## Directory Structure

```
deployment/
├── docker/               # Docker configurations
│   └── nginx.conf        # Reverse proxy config
├── scripts/             # Helper scripts
│   ├── start.sh         # Start all services
│   ├── stop.sh          # Stop all services
│   └── migrate.sh       # Database migrations
└── terraform/           # (optional) Cloud provisioning
```

## Quick Start

```bash
# Local deployment with Docker
docker-compose up --build -d

# Run training pipeline
python -m src.train_pipeline --mode full
```

## Cloud Deployment

See `docs/deployment_guide.md` for AWS, GCP, and Azure instructions.
