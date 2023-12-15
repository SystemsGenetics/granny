Installation
============

Want to try on your own dataset? You can install our model to get started. First, it is recommended to use a package
manager such as `conda <https://www.anaconda.com/>`_ or `virtualenv <https://pypi.org/project/virtualenv/>`_ to create a seperate, independent environment for **Granny**. An description of the package installation using conda is provided below.

Granny uses `this Mask R-CNN module <https://github.com/matterport/Mask_RCNN/tree/v2.1>`_, written in
TensorFlow 1.15. Due to the TensorFlow limitation, it is required to have Python version be **less than or equal** to
3.7::

    conda create -n <venv> python==3.7 -y

where ``<venv>`` is the name of the virtual environment. 
To activate the environment::

    conda activate <venv>

Now that you are inside the environment, run the following to set up command line interfaces::

    pip install --upgrade granny

Notes
-----
More ways to install **Granny** is coming!!! 
