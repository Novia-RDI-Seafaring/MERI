![alt text](media/meri_logo.svg)
## Setup
Easiest way to ensure correct setup is to run the project in a docker container. We provide a dockerfile (```docker/Dockerfile```) for this purpose. 

### docker in vscode
1. Install the following extensions in VSCode:
    - Docker
    - Dev Containers

2. Add a .devcontainer folder and create a devcontainer.json file. VSCode uses this file to build and connect to docker containers. Example devcontainer.json
    ```json
    {
        "name": "python3",
        "build": {
            "dockerfile": "../docker/Dockerfile"
        },
        // Use 'forwardPorts' to make a list of ports inside the container available locally.
        "forwardPorts": [7860],

        // Use 'postCreateCommand' to run commands after the container is created.
        "postCreateCommand": "conda env create -f environment.yaml",
        "runArgs": ["--gpus=all"],
        "remoteUser": "root"
    }
    ```
3. Press STRG + SHIFT + P and select "Dev Container: Open Folder in Container". This will build the docker container and connect the workspace to it.

4. activate the conda environment ```conda activate layout_analysis```

5. Install meri as package ```pip install -e .```

