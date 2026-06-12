FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CUDA-enabled torch first (default PyPI index ships the CPU-only wheel)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY skills/ skills/
# Bake in the pre-generated splits so jobs don't need to regenerate them
COPY data/splits/ data/splits/
COPY docker/run_job_base_eval.py run_job_base_eval.py
COPY docker/run_job_sft.py run_job_sft.py
COPY docker/run_job_skill.py run_job_skill.py

CMD ["python3", "-c", "print('Thesis model image ready. Specify a run_job_*.py entrypoint.')"]
