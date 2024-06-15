# PrismaML Installation Guide

## Table of Contents

- [User Installation Guide](#user-installation-guide)
- [Setting up Development Environment](#setting-up-your-development-environment)

## User Installation Guide

To install PrismaML, use pip:

```bash
pip install prisma_ml
```

Or using Poetry

```bash
poetry add prisma_ml
```

## Setting Up Your Development Environment

For contributors, setting up your development environment is crucial. Follow the following instructions:

1. **Clone the Repository**:

```bash
git clone https://github.com/yourusername/PrismaML.git
cd PrismaML
```

2. **Install Poetry**:
   Ensure you have Poetry installed. If not, install it using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Install Dependencies**:
   Use Poetry to install all dependencies specified in the pyproject.toml file:

```bash
poetry install
```

4. **Activate the Virtual Environment**:
   Activate the virtual environment created by Poetry:

```bash
poetry shell
```
