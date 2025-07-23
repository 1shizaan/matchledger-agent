# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for psycopg2
# These are common for Python applications interacting with C libraries like psycopg2 depends on
# Original line (likely Step 3 in your build output)
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    # Add these new dependencies for WeasyPrint:
    libcairo2 \
    libpango1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    # 'libglib2.0-0' is usually pulled in by the above, but can be added explicitly if needed.  
    # It provides libgobject-2.0-0
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    # Then, install the rest of your requirements
    pip install --no-cache-dir -r requirements.txt
COPY . .

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Command to run your application using Uvicorn
# The --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]