![alt text](media/meri_logo.svg)

Package for parameter extraction from pdf documents.
# Installation
Requirements:
- python 3.12
- CUDA 12.1 (for GPU acceleration)
## Docker
Easiest way to ensure correct setup is to run the project in a docker container. We provide a dockerfile (```docker/Dockerfile```) for this purpose. 

### manual build

1. build image: ```docker build -t meri_image -f /docker/Dockerfile .```
2. run container: ```docker run -it --gpus=all -p 7860:7860 --name meri_container meri_image```


### docker container in vscode
1. Install the following extensions in VSCode:
    - Docker
    - Dev Containers

2. Press STRG + SHIFT + P and select "Dev Container: Open Folder in Container" (devcontainer.json exists in .devcontainer). This will build the docker container and connect the workspace to it.

4. activate the conda environment ```conda activate layout_analysis```

5. Install meri as package ```pip install -e .```

# Usage
Examples on how to use the package can be found in can be found in ```docs/notebooks```

# Method
![alt text](media/meri.png)

The proposed method requires two inputs: (1) the pdf document and (2) a json schema. Our method will initially detect layout elements such as tables, text and figures. Depending on the layout type different information extraction methods are applied to structurize the content and create an intermediate format (markdown). 
The intermediate format alongside the json schema are processed by an LLM to populate the extract the specified parameters. The output will follow the provided json schema.