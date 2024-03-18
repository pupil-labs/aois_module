====================================================================
Pupil Labs AOIs Module: Automate AOI Definition with GroundingSAM
====================================================================

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. image:: https://img.shields.io/badge/skeleton-2024-informational
   :target: https://blog.jaraco.com/skeleton

.. image:: https://img.shields.io/badge/python-3.10-blue.svg
   :target: https://www.python.org/downloads/release/python-3100/
   :alt: Python version: 3.10

**Introduction**
----------------

The Pupil Labs AOIs Module leverages the powerful GroundingSAM framework to automate the process of defining Areas of Interest (AOIs) within the Reference Image Mapper/Marker Mapper enrichment in Cloud. This innovative tool offers a user-friendly web interface, allowing for the upload of reference images and the specification of desired segmentation parameters. By utilizing `GroundingSAM <https://github.com/IDEA-Research/Grounded-Segment-Anything>`_ for segmentation, and facilitating mask submission through our `Cloud API <https://api.cloud.pupil-labs.com/v2>`_, this module significantly streamlines the enrichment process, eliminating the need for manual mask drawing.


**Quick Start with Google Colab**
---------------------------------

.. image:: https://img.shields.io/static/v1?label=&message=Open%20in%20Google%20Colab&color=blue&labelColor=grey&logo=Google%20Colab&logoColor=#F9AB00
   :target: https://colab.research.google.com/drive/1SJQS6-P56wpDxJTNfZeuzwZADKK9h6ri?usp=sharing
   :alt: Open in Google Colab

**Local Installation**
----------------------

**Docker Installation**

1. Clone the repository and build the Docker image:

.. code:: bash

    git clone https://github.com/pupil-labs/aois_module
    cd aois_module
    docker build -t pupil-labs-aois-module .

**Python Package Installation**

Optional step to set up a virtual environment using UV (from astral.sh):

.. code:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv --seed --python3.11
    source .venv/bin/activate
    uv pip install pupil-labs-aois-module

Alternatively, install directly with pip:

.. code:: bash

    pip install pupil-labs-aois-module

Or, for a development setup, clone the repository and install:

.. code:: bash

    git clone https://github.com/pupil-labs/aois-module.git
    cd aois-module
    pip install .

**Running the Module**
----------------------

**Locally**

.. code:: bash

    pl-aois

**Using Docker**

.. code:: bash

    docker run -p 8002:8002 pupil-labs-aois-module
