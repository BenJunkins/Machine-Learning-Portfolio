FROM python:3.12-slim

# The first label links the published image to this repo on GitHub.
# The second is the description shown on the image's page
LABEL org.opencontainers.image.source="https://github.com/BenJunkins/Machine-Learning-Portfolio" \
      org.opencontainers.image.description="Spam filter web API: a TF-IDF and Random Forest model served with FastAPI"

# Don't write .pyc files, and print log lines straight away instead of buffering them
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the requirements before copying the code, so this slow step is
# reused from cache whenever only the code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements-api.txt

COPY spam_filter/ spam_filter/
COPY models/ models/

# Run the server as an ordinary user instead of root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# 0.0.0.0 makes the server reachable from outside the container
CMD ["uvicorn", "spam_filter.api:app", "--host", "0.0.0.0", "--port", "8000"]
