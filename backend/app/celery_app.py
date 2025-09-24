# backend/app/celery_app.py

import os
from celery import Celery

# Read Redis URL from environment variables, with a local default
broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Initialize Celery
celery = Celery(
    __name__,
    broker=broker_url,
    backend=result_backend
)

# Optional: You can include the tasks module here if you like
celery.autodiscover_tasks(['app.tasks'])

# Optional Celery configuration
celery.conf.update(
    task_track_started=True,
)