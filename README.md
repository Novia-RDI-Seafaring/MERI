![alt text](media/meri_logo.svg)

Package for parameter extraction from pdf documents. Provided with a pdf file and json schema, MERI will return a populated dictionary following the provided json schema.

## Table of Contents
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Installation from Source](#installation-from-source)
- [Docker](#docker)
    - [Manual Build](#manual-build)
    - [Docker Container in VSCode](#docker-container-in-vscode)
- [Usage](#usage)
- [Demo](#demo)
- [Method](#method)

# Installation

Requirements:
- python 3.12
- PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation.
- create .env file in workspace and place open ai key there ```OPENAI_API_KEY='sk-...'```

Installation:
- ```pip install meri @ git+https://github.com/Novia-RDI-Seafaring/MERI/tree/main ```

Installation from source:
- ```git clone git@github.com:Novia-RDI-Seafaring/MERI.git¨```
- ```pip install .``` for edible mode ```pip install -e .```


# Docker
Easiest way to ensure correct setup is to run the project in a docker container. We provide a dockerfile (```docker/Dockerfile```) for this purpose. 

### manual build

1. build image: ```docker build -t meri_image -f /docker/Dockerfile .```
2. run container: ```docker run -it --gpus=all -p 7860:7860 --name meri_container meri_image```
3. install the package inside the container following the instructions [Installation](#installation)

### docker container in vscode
1. Install the following extensions in VSCode:
    - Docker
    - Dev Containers

2. Press STRG + SHIFT + P and select "Dev Container: Open Folder in Container" (devcontainer.json exists in .devcontainer). This will build the docker container and connect the workspace to it.

3. Install meri following the instructions [Installation](#installation)

# Usage
Examples on how to use the package can be found in can be found in ```docs/notebooks```

# Demo
We provide a gradio demo in ```demo```. Run ```python demo/demo_meri_v1.py```. In ```data/demo_data``` we provide a example data sheet alongside a dummy json schema that specifies the parameters of interest. Upload both and run the extraction pipeline.


# Method
![alt text](media/meri.png)

The proposed method requires two inputs: (1) the pdf document and (2) a json schema. Our method will initially detect layout elements such as tables, text and figures. Depending on the layout type different information extraction methods are applied to structurize the content and create an intermediate format (markdown). 
The intermediate format alongside the json schema are processed by an LLM to populate the extract the specified parameters. The output will follow the provided json schema.