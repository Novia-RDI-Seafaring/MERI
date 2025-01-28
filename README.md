[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Novia-RDI-Seafaring/MERI/badge)](https://scorecard.dev/viewer/?uri=github.com/Novia-RDI-Seafaring/MERI)

 ![alt text](media/meri_logo.svg)

Package for parameter extraction from pdf documents. Provided with a pdf file and json schema, MERI will return a populated dictionary following the provided json schema.

## Table of Contents
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Installation from Source](#installation-from-source)
- [Docker](#docker)
    - [Development in docker](#development-in-docker)
    - [Run MERI demo in docker](#run-meri-demo-in-docker)
- [Usage](#usage)
    - [MERI](#meri-class)
    - [LLM](#llms)
- [Demo](#demo)
- [Method](#method)


# Installation

Requirements:
- software is tested with python 3.12
- poetry package manager
- create .env file in workspace and place the respective variables there (see section [LLM](#llms))

Installation:
- ```pip install meri @ git+https://github.com/Novia-RDI-Seafaring/MERI/tree/main ```. 

Installation from source:
- ```git clone git@github.com:Novia-RDI-Seafaring/MERI.git¨```
- ```pip install .``` for edible mode ```pip install -e .```


# Docker

Easiest way to ensure correct setup is to run the project in a docker container. We provide two dockerfiles (```docker/```).

1. dev.Dockerfile installs all dependencies and can be used as a devcontainer in vscode [Development in docker](#development-in-docker)
2. app.Dockerfile installs all dependencies and runs the meri demo that is accessible via the browser on localhost:5010 [Run meri in docker](#run-meri-in-docker)

## Development in docker
1. Install the following extensions in VSCode:
    - Docker
    - Dev Containers

2. Press STRG + SHIFT + P and select "Dev Container: Open Folder in Container" (devcontainer.json exists in .devcontainer). This will build the docker container and connect the workspace to it.


## Run MERI demo in docker
To run MERI gradio demo in docker and forward the respective port:
1. build image: ```docker build -t meri_app -f /docker/app.Dockerfile .```
2. run container: ```docker run -it --gpus=all -p 5010:5010 --name meri_app_container meri_app```

Easiest way to ensure correct setup is to run the project in a docker container. We provide a dockerfile (```docker/Dockerfile```) for this purpose. 

# Usage

## MERI Class

The `MERI` class is designed for parameter extraction from PDF documents. It takes several arguments that configure its behavior.

### Arguments

- **`pdf_path`** (`str`): The path to the PDF file from which parameters will be extracted.
  
- **`chunks_max_characters`** (`int`, optional): Threshold for chunking the intermediate format. default 450000.

- **`model`** (`str`, optional): Name of the model that is to be used, following the naming of LiteLLM framework. default: gpt-4o-mini

- **`model_temp`** (`str`, optional): Model temperature. default: 0.0.

- **`do_ocr`** (`bool`, optional): Docling configuration. If false the native pdf text is used. If true, ocr is applied to extract the text. default: false.

- **`do_cell_matching`** (`bool`, optional): Refinment of cell detection by layout model through cell matching.

### Example Usage

```python 
from meri import MERI
import json

pdf_path ='path/to/pdf.pdf'

# must be a valid json schema
schema_path ='path/to/schema.json'
with open(schema_path) as f:
    schema = json.load(f)

meri = MERI(pdf_path=pdf_path)

# populate provided json schema
populated_schema = meri.run(json.dumps(schema))

```


More examples how to use the package can be found in can be found in ```docs/notebooks```

### LLMs
This package uses LiteLLM as a wrapper to interact with LLMs. The model name can be provided as parameter to MERI and the required environment variable must be set in the .env file. 
- OpenAI API: provide OPENAI_API_KEY in the .env file. Model name will be e.g. gpt-4o-mini
- Azure API: provide AZURE_API_KEY and AZURE_API_BASE in the .env and the model name will be e.g. azure/gpt-4o

The models must be multi-modal model, i.e. be able to process text as well as images.

# Demo
We provide a gradio demo in ```demo```. Run ```poetry run python app/app.py --model gpt-4o-mini```. In ```data/demo_data``` we provide a example data sheet alongside a dummy json schema that specifies the parameters of interest. Upload both and run the extraction pipeline.

# Method
![alt text](media/meri.png)

Parameter extraction from documents with MERI follows the two-step approach: 

(1) Layout elements, such as text, tables, and figures, are detected and individually processed to create an intermediate machine-readable representation of the whole document. 

(2) The intermediate format, along with the task description (prompt and blueprint), is processed by an LLM that outputs a populated version of the blueprint containing found parameters and their attributes. Below is a more in-depth explanation of the steps and formats involved.


## Acknowledgments

This work was done in the Business Finland funded project [Virtual Sea Trial](https://virtualseatrial.fi).

## License

This package is licensed under the MIT License license. See the [LICENSE](./LICENSE) file for more details.

## Citation

If you use this package in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{MERI,
  author = {Christian Möller, Lamin Jatta},
  title = {MERI: Modality-Aware Extraction and Retrieval of Information},
  year = {2024},
  howpublished = {\url{https://github.com/Novia-RDI-Seafaring/MERI}},
}
