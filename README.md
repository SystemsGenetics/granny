"**GRANNY** is going to rate your apples"
---
**GRANNY** is a program for superficial disorder rating in "Granny Smith" apples. 

## Introduction 

Superficial scald is a physiological disorder that occurs following a chilling injury during early weeks of fruit storage, but with delayed symptom development - most notably peel necrosis that occurs in irregular patterns. Currently, quantification of the superficial scald incidence is performed manually by trained technicians with a few limitations: 
- The set of rating values is small and coarse-grained. 
- The ratings are subjected to human error and individual bias
- 

In order to resolve the problem, the [Ficklin Research Program](http://ficklinlab.cahnrs.wsu.edu/) at Washington State University has developed a computer vision approach for automatic detection and rating in apple images, specifically the "Granny Smith" cultivar.

Inspired by the well-known apple cultivar, **GRANNY** is Python-based implementation of Mask-RCNN and image segmentation techniques, aiming to assist technicians in the superficial disorder rating process. 

Usage guidelines of **GRANNY** are provided. 

## Installation and Usage

### Installation

It is recommended to use a package manager such as [conda](https://www.anaconda.com/) or [virtualenv](https://pypi.org/project/virtualenv/) to create a seperate, independent environment for **GRANNY**. 

To create a virtual environment using conda:
```bash
    conda create -n <venv>
```
Where `<venv>` is the name of the virtual environment

To activate the environment:
```bash
    conda activate <venv>
```

Due to the limitation of TensorFlow 1.15, it is required to have Python version be **less than or equal** to 3.7
```bash
    conda install python==3.7
```

To install **GRANNY**, use the following:
```bash
    pip install granny
```

### Command Line Arguments
To perform extraction and/or rating on "Granny Smith" images, run **GRANNY** in the following way in the command line: 

```bash
    granny  [-a,--action] <ACTION>  [-p,--image_path] <PATH>  [-m,--mode] <MODE>  [-v,--verbose] <VERBOSE>
```

where: 
| Arguments  | Type | Values |
| ---------  | ---- | ------ |
| `<ACTION>` | Required | Either "extract" or "rate" <br />"extract" - perform individual apples extraction from apple trays; <br /> "rate" - perform disorder rating on single apple images.|
| `<PATH>`   | Required | Specify either an image directory containing multiple images or a single image file. |
| `<MODE>`   | Optional | Default to 1. Specify 2 for multiple images processing in `--action rate`. |
| `<VERBOSE>` | Optional | Default to 1. Specify 0 to turn off model display. |

### Output

### **GRANNY** Directory
Please refer to [**GRANNY** directory description](https://github.com/SystemsGenetics/granny/blob/master/GRANNY/README.md) for a detailed listing of the **GRANNY**'s components.  

### Limitations 
This program contains limitations. 



