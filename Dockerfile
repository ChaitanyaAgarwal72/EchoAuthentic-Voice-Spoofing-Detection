FROM python:3.10-slim

RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    nodejs \
    && ln -s /usr/bin/nodejs /usr/bin/node || true \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user backend/requirements.txt $HOME/app/

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user backend/ $HOME/app/

USER user

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
