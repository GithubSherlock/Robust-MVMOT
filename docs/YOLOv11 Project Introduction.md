## 🚀 YOLOv11 Project Introduction

## Construction analysis for project catalog

### 2.1 .github

…

### 2.2 docker

Contains multiple Docker profiles to support deployment in different environments or platforms:

* **Dockerfile**: The default Docker image configuration.
* **Dockerfile-arm64**: Adapts to ARM64 architecture devices.
* **Dockerfile-conda**: Environment configuration based on the Conda package manager.
* **Dockerfile-cpu**: Customised configuration for CPU environments.
* **Dockerfile-jetson**: Designed for the NVIDIA Jetson platform.
* **Dockerfile-python**: Simplified configuration for Python-only environments.
* **Dockerfile-runner**: May be used in CI/CD environments.

**Function**: Provides users with flexible deployment options.

### 2.3 docs

Stores documentation materials, including translations in multiple languages and documentation build tools:

`build_docs.py`: used to automate the generation of documents.

With `mkdocs` and other tools to achieve document management.

### 2.4 examples

Provide sample code for YOLOv11 to help users get started quickly.

### 2.5 tests

Contains test code to verify the functionality and stability of the project.

### 2.6 ultralytics

#### 2.6.1 assets

Two very classic images used for testing in YOLO history.

#### 2.6.2 cfg

#### 2.6.3 data

### 2.7 Files in sibling directories