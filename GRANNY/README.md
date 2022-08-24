GRANNY is a Python-based implementation of Mask-RCNN and image segmentation technique to generate ratings for superficial disorders in "Granny Smith" apples. 

## Introduction 

Superficial scald is a physiological disorder that occurs following a chilling injury during early weeks of fruit storage, but with delayed symptom development - most notably peel necrosis that occurs in irregular patterns. Currently, quantification of the superficial scald incidence is performed manually by trained technicians, often using a small set of rating values. Human error and individual bias, and the coarse-grained rating scale can lead to inconsistencies in estimation of disorder severity. 


### GRANNY Directory 

| File Name | Purpose |
| -------------- | ------- |
| `GRANNY.py` | Performs image extraction/ scald rating on Granny Smith apple images. |
| `GRANNY_config.py` | Configures Mask-RCNN to use for GRANNY. |
| `command.py` | Sets up to run GRANNY using the command line. |


## Installation and Usage

### Installation

It is recommended to use a package manager such as [conda](https://www.anaconda.com/) or [virtualenv](https://pypi.org/project/virtualenv/) to create a seperate, independent environment. 

To create a virtual environment using conda:
```bash
    conda create -n <venv>
```
Where `<venv>` is the name of the virtual environment

To activate the environment:
```bash
    conda activate <venv>
```

Due to the limitation of TensorFlow 1.15, it is required to have Python version be less than or equal to 3.7
```bash
    conda install python==3.7
```

To install GRANNY, use the following:
```bash
    pip install granny
```

### Command Line Arguments
To perform extraction and rating on "Granny Smith" images, run granny in the following way in the command line: 

```bash
    granny --action <ACTION> --image_path <PATH> --mode <MODE> --verbose <VERBOSE>
```

Where: 
| Arguments  | Type | Values |
| ---------  | ---- | ------ |
| `<ACTION>` | Required | Specify "extract" is to perform individual apples extraction from apple trays. "rate" is to perform disorder rating on single apple images.|
| `<PATH>`   | Required | Specify either an image directory containing multiple images or a single image file. |
| `<MODE>`   | Optional | Specify 2 for multiple images processing. |
| `<VERBOSE>` | Optional | Specify 0 to turn off model display. |

### Limitations 
