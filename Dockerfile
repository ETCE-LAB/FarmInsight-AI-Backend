FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps needed by LightGBM (libgomp.so.1) + certificates
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY ./model_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of your Django project
COPY ./model_service /app/

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
