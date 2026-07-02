# SkillLens AI API Reference

## Base URL

Local: `http://localhost:8000`
Production: `https://your-domain.com`

## Authentication

Currently, the API does not require authentication for local use. For production deployments, configure API key authentication.

## Endpoints

### Root

`GET /`

Returns API information.

#### Response

```json
{
    "app": "SkillLens AI",
    "version": "1.0.0",
    "status": "running"
}
```

### Health Check

`GET /health`

Returns service health status.

#### Response

```json
{
    "status": "healthy"
}
```

### Extract Skills from Text

`POST /api/v1/skills/extract`

Extract skills from job description or resume text.

#### Request Body (Form Data)

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Text to extract skills from |

#### Response

```json
{
    "skills": ["Python", "Machine Learning", "SQL"],
    "count": 3
}
```

### Get Skill Trends

`POST /api/v1/skills/trends`

Get demand trends for a specific skill.

#### Request Body (JSON)

| Field | Type | Description |
|-------|------|-------------|
| `skill_name` | string | Name of the skill |

#### Response

```json
{
    "skill": "Python",
    "trend": "increasing",
    "growth_rate": 25.5,
    "projected_demand": 92
}
```

### Predict Salary

`POST /api/v1/salary/predict`

Predict salary based on job details.

#### Request Body (JSON)

| Field | Type | Description |
|-------|------|-------------|
| `job_title` | string | Job title (e.g., "Data Scientist") |
| `years_experience` | float | Years of experience |
| `location` | string | Job location |
| `education` | string | Education level (Bachelor, Master, PhD) |
| `skills` | array[string] | Optional list of skills |

#### Response

```json
{
    "predicted_salary": 190440.0,
    "confidence_interval": {"lower": 161874.0, "upper": 219006.0},
    "factors": [
        {"name": "Job Title", "impact": "high", "value": "Data Scientist"},
        {"name": "Experience", "impact": "high", "value": 5},
        {"name": "Location", "impact": "medium", "value": "San Francisco"},
        {"name": "Education", "impact": "medium", "value": "Master"}
    ]
}
```

### Analyze Resume

`POST /api/v1/resume/analyze`

Upload resume for ATS scoring and skill extraction.

#### Request (Multipart Form)

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Resume file (PDF, DOCX, or TXT) |
| `job_description` | string | Optional job description for matching |

#### Response

```json
{
    "ats_score": 78.5,
    "extracted_skills": ["Python", "Machine Learning", "SQL"],
    "missing_skills": ["Kubernetes", "Airflow"],
    "suggestions": ["Add more relevant skills from the job description"]
}
```

### Forecast Skill Demand

`POST /api/v1/forecast`

Forecast demand for a specific skill.

#### Request Body (JSON)

| Field | Type | Description |
|-------|------|-------------|
| `skill_name` | string | Name of the skill |
| `periods` | int | Number of months to forecast (default: 12) |

#### Response

```json
{
    "skill": "Python",
    "forecast": [
        {"date": "2025-01", "value": 70.0},
        {"date": "2025-02", "value": 72.0}
    ],
    "growth_rate": 14.3,
    "trend": "up"
}
```

### Analyze Skill Gap

`POST /api/v1/skill-gap`

Analyze skill gap between user skills and market demand.

#### Request Body (JSON)

| Field | Type | Description |
|-------|------|-------------|
| `skills` | array[string] | User's current skills |
| `target_role` | string | Optional target job role |

#### Response

```json
{
    "matched_skills": [...],
    "missing_skills": [...],
    "match_percentage": 22.2,
    "overall_gap_score": 77.8
}
```

### Chat with AI Assistant

`POST /api/v1/chat`

Query the RAG-based career assistant.

#### Request Body (JSON)

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | User's question |
| `history` | array | Optional conversation history |

#### Response

```json
{
    "response": "Top skills for 2025 include Generative AI, MLOps...",
    "confidence": 0.67,
    "sources": [...]
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad request (invalid input) |
| 422 | Validation error (missing required fields) |
| 500 | Internal server error |

## Rate Limiting

Not currently implemented. For production, configure rate limiting at the reverse proxy level.
