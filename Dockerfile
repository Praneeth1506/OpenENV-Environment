FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    transformers \
    "trl>=0.9.0" \
    datasets \
    peft \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    matplotlib \
    numpy

# Copy repo
COPY . .

EXPOSE 7860

# space_entrypoint.py runs training + serves live logs on 7860
CMD ["python", "space_entrypoint.py"]
