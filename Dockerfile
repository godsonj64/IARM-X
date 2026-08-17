FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e .
CMD ["python", "scripts/smoke_test.py"]
