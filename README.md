## "**Granny** is going to rate your apples"

[![Documentation Status](https://readthedocs.org/projects/granny/badge/?version=latest)](https://granny.readthedocs.io/en/latest/?badge=latest)


**Granny** is a computer vision software aiming to assist technicians in post-harvest fruit maturity index experiments.

---

## Introduction

Superficial scald is a physiological disorder that occurs following a chilling injury during early weeks of fruit storage, but with delayed symptom development - most notably peel necrosis that occurs in irregular patterns. Currently, quantification of the superficial scald incidence is performed manually by trained technicians with a few inconsistencies:

- The set of rating values is small and coarse-grained.
- The ratings are subjected to human error and individual bias

In the collaboration with [Honaas lab](https://www.ars.usda.gov/pacific-west-area/wenatchee-wa/physiology-and-pathology-of-tree-fruits-research/people/loren-honaas/) at the USDA ARS, the [Ficklin Research Program](http://ficklinlab.cahnrs.wsu.edu/) at Washington State University has developed a computer vision approach for automatic detection and superficial scald rating in the "Granny Smith" apple images.

Inspired by the well-known apple cultivar, **Granny** is Python-based implementation of Mask-RCNN and image segmentation techniques, aiming to assist technicians in post-harvest maturity index experiments.

---

## Project Contributors

- [Nhan Nguyen](http://ficklinlab.cahnrs.wsu.edu/#people) - Ficklin lab's undergraduate researcher
- [Joseph Michaud](https://www.ars.usda.gov/people-locations/person/?person-id=57281) - Biological Science Technician, USDA-ARS Hood River
- [Heidi Hargarten](https://www.ars.usda.gov/people-locations/person?person-id=52227) - Honaas lab's postdoctoral researcher, USDA-ARS Wenatchee
- [Loren Honaas](https://www.ars.usda.gov/pacific-west-area/wenatchee-wa/physiology-and-pathology-of-tree-fruits-research/people/loren-honaas/) - Honaas lab's principal investigator
- [Stephen Ficklin](http://ficklinlab.cahnrs.wsu.edu/) - Ficklin lab's principal investigator

---

## Usage

The main usage of the program consists of 2 main steps:

- Step 1. Locate and extract instances using Mask-RCNN _[Instance Extraction]_.
- Step 2. Rate each instance. Depending on your purpose, you can tell the program to
  - rate superficial scald in Granny Smith apples _[Superficial Scald]_
  - calculate starch area on iodine-stained cross-sections _[Cross-section Starch]_
  - sort color of the pears _[Pear Color]_

Please refer to our [demo](https://github.com/SystemsGenetics/granny/tree/master/demo) page for a walk-through of each step's input and output.

---

## Installation

Please refer to our [docs](https://github.com/SystemsGenetics/Granny/tree/master/docs) page for a detailed explanation of how to install the program and related packages in Python.

---

## Limitations

We are sorry in advance for the inconvenience, but our program still contains a few limitations.

- _[Instance Extraction]_ Due to the dependency requirements, it is recommended to install Python's package managers such as [conda](https://www.anaconda.com/) or [virtualenv](https://pypi.org/project/virtualenv/). Here, we provide instructions for installation of packages using conda.
- _[Superficial Scald]_ Due to the similarity in coloration, the stem of the apples could be potentially counted towards the total area of superficial scald.
- _[Cross-section Starch]_ In order to run our provided ImageJ scripts for starch analysis, the user must seperately install [Fiji](https://imagej.net/software/fiji/). We are currently trying to implement the Python wrapper function for calling ImageJ macros.
- _[Cross-section Starch]_ Due to the similarity in coloration, the code is unable to distinguish between bruising and iodine-stained area.
- _[Pear Color]_ Currently, the color of single pear image is only being sorted based on the provided color card under [demo/pear_images/color_preference](). 
