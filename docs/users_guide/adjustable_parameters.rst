Adjustable Parameters
=====================

This guide covers the configurable parameters available in Granny for fine-tuning analysis results.

Segmentation Parameters
-----------------------

Analysis Parameters
~~~~~~~~~~~~~~~~~~~

Confidence Threshold (``--confidence``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 0.25 | **Range:** 0.0 - 1.0 | **Type:** Float
- Controls YOLO detection confidence. Higher values = fewer, more accurate detections.

IOU Threshold (``--iou_threshold``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 0.45 | **Range:** 0.0 - 1.0 | **Type:** Float
- Intersection over Union threshold for non-maximum suppression. Controls how overlapping detections are filtered.

Row Grouping Tolerance (``--row_tolerance``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 20 | **Range:** 1 - 100 | **Type:** Integer
- Controls fruit row detection sensitivity. Fruits grouped into rows when y-centers differ by more than height/row_tolerance pixels.

Visualization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Mask Alpha (``--mask_alpha``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 0.5 | **Range:** 0.0 - 1.0 | **Type:** Float
- Transparency of mask overlay on output images (0.0 = transparent, 1.0 = opaque).

Color Brightness (``--color_brightness``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 1.0 | **Range:** 0.0 - 1.0 | **Type:** Float
- Brightness value for mask colors in HSV color space.

Bounding Box Thickness (``--bbox_thickness``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 5 | **Range:** 1 - 50 | **Type:** Integer
- Thickness of bounding box lines in pixels.

Font Scale (``--font_scale``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 2.0 | **Range:** 0.1 - 10.0 | **Type:** Float
- Font scale for text labels on output images.

Text Thickness (``--text_thickness``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 3 | **Range:** 1 - 50 | **Type:** Integer
- Thickness of text labels in pixels.

----

Starch Analysis Parameters
---------------------------

Analysis Parameters
~~~~~~~~~~~~~~~~~~~

Starch Threshold (``--starch_threshold``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 172 | **Range:** 0 - 255 | **Type:** Integer
- Pixels with gray values ≤ this threshold are considered starch. Lower = only darkest regions, higher = includes lighter regions.

Blur Kernel (``--blur_kernel``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 7 | **Range:** 1 - 99 (odd values recommended) | **Type:** Integer
- Gaussian blur kernel size for noise reduction. Larger values = more smoothing.

Visualization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Mask Alpha (``--mask_alpha``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 0.6 | **Range:** 0.0 - 1.0 | **Type:** Float
- Transparency of starch mask overlay on output images.

----

Blush Color Analysis Parameters
--------------------------------

Analysis Parameters
~~~~~~~~~~~~~~~~~~~

Blush Threshold (``--threshold``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 148 | **Range:** 0 - 255 | **Type:** Integer
- A channel threshold in LAB color space for blush detection.

Fruit Threshold (``--fruit_threshold``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 140 | **Range:** 0 - 255 | **Type:** Integer
- B channel threshold in LAB color space for fruit pixel detection.

Visualization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Blush Color RGB (``--blush_color_r``, ``--blush_color_g``, ``--blush_color_b``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Defaults:** R=150, G=55, B=50 | **Range:** 0 - 255 each | **Type:** Integer
- RGB color values for blush mask overlay (BGR format).

Text Position (``--text_x``, ``--text_y``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Defaults:** X=20, Y=50 | **Range:** 0 - 5000 | **Type:** Integer
- X and Y coordinates for text position on output images.

Font Scale (``--font_scale``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 1.0 | **Range:** 0.1 - 10.0 | **Type:** Float
- Font scale for text labels.

Text Thickness (``--text_thickness``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 3 | **Range:** 1 - 50 | **Type:** Integer
- Thickness of text labels in pixels.

----

Superficial Scald Analysis Parameters
--------------------------------------

Analysis Parameters
~~~~~~~~~~~~~~~~~~~

Morphological Kernel (``--morph_kernel``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 10 | **Range:** 1 - 99 | **Type:** Integer
- Size of ellipse kernel for morphological operations. Larger = more smoothing.

Minimum Threshold (``--min_threshold``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 100 | **Range:** 0 - 255 | **Type:** Integer
- Minimum threshold for scald detection. Pixels below this are potential scald regions.

Purple Threshold (``--purple_threshold``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 126 | **Range:** 0 - 255 | **Type:** Integer
- Threshold for removing purple background/tray pixels using YCrCb color space.

Blur Kernel (``--blur_kernel``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 3 | **Range:** 1 - 99 (odd values recommended) | **Type:** Integer
- Gaussian blur kernel size for image smoothing.

Histogram Factor (``--hist_factor``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 0.333 | **Range:** 0.0 - 1.0 | **Type:** Float
- Fraction of histogram range to subtract from threshold calculation.

Histogram Top N (``--hist_top_n``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 10 | **Range:** 1 - 100 | **Type:** Integer
- Number of top histogram values to consider for threshold calculation.

----

Peel Color Analysis Parameters
-------------------------------

Analysis Parameters
~~~~~~~~~~~~~~~~~~~

Purple Threshold (``--purple_threshold``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 126 | **Range:** 0 - 255 | **Type:** Integer
- Threshold for removing purple background/tray pixels using YCrCb color space.

Lightness Range (``--lightness_min``, ``--lightness_max``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Defaults:** Min=0, Max=255 | **Range:** 0 - 255 | **Type:** Integer
- Lightness channel range in LAB color space for peel color detection.

Green Range (``--green_min``, ``--green_max``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Defaults:** Min=0, Max=128 | **Range:** 0 - 255 | **Type:** Integer
- Green channel range in LAB color space for green detection.

Yellow Range (``--yellow_min``, ``--yellow_max``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Defaults:** Min=128, Max=255 | **Range:** 0 - 255 | **Type:** Integer
- Yellow channel range in LAB color space for yellow detection.

Normalize Lightness (``--normalize_lightness``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Default:** 50 | **Range:** 0 - 100 | **Type:** Integer
- Target lightness value for color normalization in LAB space.

----

Usage Examples
--------------

Segmentation with Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   granny -i cli --analysis segmentation --input ./images/ --confidence 0.35 --iou_threshold 0.6 --row_tolerance 15

Starch Analysis with Custom Threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   granny -i cli --analysis starch --input ./segmented/ --starch_threshold 180 --blur_kernel 9

Blush with Custom Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   granny -i cli --analysis blush --input ./pears/ --threshold 150 --blush_color_r 200 --font_scale 1.5

Scald with Fine-Tuned Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   granny -i cli --analysis scald --input ./apples/ --min_threshold 105 --hist_factor 0.4 --hist_top_n 15

Peel Color with Custom Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   granny -i cli --analysis color --input ./pears/ --green_max 125 --yellow_min 130 --normalize_lightness 55

----

Tips
----

- **Start with defaults** and adjust incrementally based on your specific images
- **Visualization parameters** only affect output images, not analysis results
- **Analysis parameters** directly impact ratings and measurements
- Test different values on a small subset before processing large batches
- Document which parameters work best for your specific fruit varieties and imaging conditions
