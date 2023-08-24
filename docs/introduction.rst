Introduction
============

Superficial scald is a physiological disorder that occurs following a chilling injury during early weeks of fruit storage, but with delayed symptom development - most notably peel necrosis that occurs in irregular patterns. Currently, quantification of the superficial scald incidence is performed manually by trained technicians with a few inconsistencies:

- The set of rating values is small and coarse-grained.
- The ratings are subjected to human error and individual bias

In the collaboration with `Honaas lab <https://www.ars.usda.gov/pacific-west-area/wenatchee-wa/physiology-and-pathology-of-tree-fruits-research/people/loren-honaas/>`_ at the USDA ARS, the [Ficklin Research Program](http://ficklinlab.cahnrs.wsu.edu/) at Washington State University has developed a computer vision approach for automatic detection and superficial scald rating in the "Granny Smith" apple images.

Inspired by the well-known apple cultivar, **Granny** is Python-based implementation of Mask-RCNN and image segmentation techniques, aiming to assist technicians in post-harvest maturity index experiments.