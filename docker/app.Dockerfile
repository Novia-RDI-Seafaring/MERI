FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install poppler-utils -y
RUN apt-get install tesseract-ocr -y

RUN pip install torch torchvision torchaudio


RUN git clone https://github.com/Novia-RDI-Seafaring/MERI.git

RUN pip install /MERI

# launch the demo
WORKDIR /MERI/demo

ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 7860

CMD ["python", "demo_meri_v1.py"]