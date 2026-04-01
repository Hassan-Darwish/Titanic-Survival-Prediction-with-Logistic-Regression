
# using python 3.10.19 same as the one used locally
FROM python:3.10.19-bookworm

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Containarize the files inside the image
COPY main.py /app/main.py
COPY trained_bias.npy /app/trained_bias.npy
COPY trained_weights.npy /app/trained_weights.npy

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
