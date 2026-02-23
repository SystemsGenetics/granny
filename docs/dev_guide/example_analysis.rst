Complete Example Analysis
=========================

This page provides a complete, working example of a fruit bruise detection analysis module. This example demonstrates all key concepts needed to create a custom analysis for Granny.

Overview
--------

This example implements a **BruiseDetection** analysis that:

- Detects bruises on fruit images using color thresholding
- Calculates bruise percentage
- Provides adjustable threshold and visualization parameters
- Outputs annotated images and CSV results
- Uses the multiprocessing architecture via ``_preRun()``, ``_processImage()``, ``_postRun()``

The algorithm uses LAB color space to detect darker regions that indicate bruising.

Complete Implementation
-----------------------

File: ``Granny/Analyses/BruiseDetection.py``

.. code-block:: python

   """
   Bruise detection analysis for fruit quality assessment.

   This module detects and quantifies bruises on fruit images by analyzing
   color differences in LAB color space. Darker regions below a threshold
   are classified as bruises.

   The analysis outputs:
   - Annotated images showing bruise regions
   - CSV file with bruise percentages for each fruit
   - Metadata including threshold values and timestamp

   Author: Example Author
   Date: 2024-01-15
   """

   import csv
   import os
   from datetime import datetime
   from typing import Dict, List, Tuple

   import cv2
   import numpy as np
   from numpy.typing import NDArray

   from Granny.Analyses.Analysis import Analysis
   from Granny.Models.Images.Image import Image
   from Granny.Models.IO.RGBImageFile import RGBImageFile
   from Granny.Models.Values.IntValue import IntValue
   from Granny.Models.Values.FloatValue import FloatValue
   from Granny.Models.Values.StringValue import StringValue
   from Granny.Models.Values.ImageListValue import ImageListValue
   from Granny.Models.Values.MetaDataValue import MetaDataValue


   class BruiseDetection(Analysis):
       """
       Analysis class for detecting and quantifying bruises on fruit images.

       This analysis processes individual fruit images and detects bruises by
       identifying darker regions in LAB color space. Results include both
       visual annotations and quantitative measurements.

       The base Analysis class handles:
       - Loading images from input directory
       - Parallel processing via multiprocessing.Pool
       - CPU core management

       You implement:
       - _preRun(): Setup before processing
       - _processImage(): Process single image (runs in parallel)
       - _postRun(): Save results after all images processed

       Attributes:
           images (List[Image]): Loaded input images (set by base class)
           input_images (ImageListValue): Input directory parameter
           output_images (ImageListValue): Output directory for annotated images
           lightness_threshold (IntValue): L-channel threshold for bruise detection
           min_bruise_area (IntValue): Minimum area (pixels) for valid bruise
           morphological_kernel (IntValue): Kernel size for noise removal
           mask_alpha (FloatValue): Transparency for bruise mask overlay
       """

       # This name will be used in CLI: granny -i cli --analysis bruise
       __analysis_name__ = "bruise"

       def __init__(self):
           """
           Initialize the bruise detection analysis with default parameters.

           Sets up input/output directories, detection thresholds, and
           visualization parameters with sensible defaults.
           """
           # STEP 1: Initialize base class (REQUIRED)
           super().__init__()

           # STEP 2: Set up input parameter
           self.input_images = ImageListValue(
               "input",
               "input",
               "Directory containing fruit images to analyze for bruises."
           )
           self.input_images.setIsRequired(True)  # User must provide this
           self.addInParam(self.input_images)

           # STEP 3: Set up output directories
           # Create timestamped output directory
           result_dir = os.path.join(
               os.curdir,
               "results",
               self.__analysis_name__,
               datetime.now().strftime("%Y-%m-%d-%H-%M")
           )

           self.output_images = ImageListValue(
               "output",
               "output",
               "Directory where annotated images will be saved."
           )
           self.output_images.setValue(result_dir)
           self.addInParam(self.output_images)

           self.output_results = MetaDataValue(
               "results",
               "results",
               "Directory where CSV results will be saved."
           )
           self.output_results.setValue(result_dir)

           # STEP 4: Define analysis parameters
           # These parameters affect the analysis results

           self.lightness_threshold = IntValue(
               "l_thresh",
               "lightness_threshold",
               "Lightness (L channel in LAB) threshold for bruise detection. "
               "Pixels with L values below this are considered bruises. "
               "Range: 0-255, default: 80."
           )
           self.lightness_threshold.setMin(0)
           self.lightness_threshold.setMax(255)
           self.lightness_threshold.setValue(80)
           self.lightness_threshold.setIsRequired(False)
           self.addInParam(self.lightness_threshold)

           self.min_bruise_area = IntValue(
               "min_area",
               "min_bruise_area",
               "Minimum area in pixels for a region to be counted as a bruise. "
               "Smaller regions are considered noise. "
               "Range: 1-10000, default: 100."
           )
           self.min_bruise_area.setMin(1)
           self.min_bruise_area.setMax(10000)
           self.min_bruise_area.setValue(100)
           self.min_bruise_area.setIsRequired(False)
           self.addInParam(self.min_bruise_area)

           self.morphological_kernel = IntValue(
               "morph_kernel",
               "morph_kernel",
               "Kernel size for morphological operations (noise removal). "
               "Larger values = more smoothing. "
               "Range: 1-99, default: 5."
           )
           self.morphological_kernel.setMin(1)
           self.morphological_kernel.setMax(99)
           self.morphological_kernel.setValue(5)
           self.morphological_kernel.setIsRequired(False)
           self.addInParam(self.morphological_kernel)

           # STEP 5: Define visualization parameters
           # These parameters only affect the visual output, not the measurements

           self.mask_alpha = FloatValue(
               "alpha",
               "mask_alpha",
               "Transparency of bruise mask overlay on output images. "
               "0.0 = transparent, 1.0 = opaque. "
               "Range: 0.0-1.0, default: 0.6."
           )
           self.mask_alpha.setMin(0.0)
           self.mask_alpha.setMax(1.0)
           self.mask_alpha.setValue(0.6)
           self.mask_alpha.setIsRequired(False)
           self.addInParam(self.mask_alpha)

           self.font_scale = FloatValue(
               "font",
               "font_scale",
               "Font scale for text annotations on output images. "
               "Range: 0.1-10.0, default: 1.0."
           )
           self.font_scale.setMin(0.1)
           self.font_scale.setMax(10.0)
           self.font_scale.setValue(1.0)
           self.font_scale.setIsRequired(False)
           self.addInParam(self.font_scale)

           # Bruise mask color (BGR format for OpenCV)
           self.bruise_color = (0, 0, 255)  # Red

       # =========================================================================
       # REQUIRED ABSTRACT METHODS
       # =========================================================================

       def _preRun(self):
           """
           Setup before image processing begins.

           Called once by performAnalysis() before parallel processing starts.
           self.images is already populated with loaded Image objects.

           Use this for:
           - Getting parameter values
           - Initializing output directory
           - Printing analysis info
           """
           # Get output directory path
           self.output_dir = self.output_images.getValue()
           self.results_dir = self.output_results.getValue()

           # Print analysis info
           print(f"\n{'='*60}")
           print(f"BRUISE DETECTION ANALYSIS")
           print(f"{'='*60}")
           print(f"Input directory:  {self.input_images.getValue()}")
           print(f"Output directory: {self.output_dir}")
           print(f"\nAnalysis Parameters:")
           print(f"  Lightness threshold: {self.lightness_threshold.getValue()}")
           print(f"  Min bruise area:     {self.min_bruise_area.getValue()} pixels")
           print(f"  Morph kernel:        {self.morphological_kernel.getValue()}")
           print(f"\nVisualization Parameters:")
           print(f"  Mask alpha:          {self.mask_alpha.getValue()}")
           print(f"  Font scale:          {self.font_scale.getValue()}")
           print(f"\nProcessing {len(self.images)} images...")
           print(f"{'='*60}\n")

       def _processImage(self, image: Image) -> Image:
           """
           Process a single image for bruise detection.

           This method runs in PARALLEL across multiple CPU cores.
           Each call receives one Image and must return the processed Image.

           IMPORTANT: Don't modify shared state here - it won't work
           with multiprocessing. Return all results via the Image object.

           Args:
               image: Input Image instance (filepath set, image not loaded)

           Returns:
               Image: The same Image with processed data and metadata
           """
           # Get parameter values (safe - these are read-only)
           l_thresh = self.lightness_threshold.getValue()
           min_area = self.min_bruise_area.getValue()
           kernel_size = self.morphological_kernel.getValue()
           alpha = self.mask_alpha.getValue()
           font_scale = self.font_scale.getValue()

           # Load the image data
           image_io = RGBImageFile()
           image_io.setFilePath(image.getFilePath())
           image.loadImage(image_io)

           # Get the numpy array (BGR format from OpenCV)
           img_array = image.getImage()

           # Perform bruise detection
           result_array, bruise_pct, bruise_count = self._detect_bruises(
               img_array, l_thresh, min_area, kernel_size, alpha, font_scale
           )

           # Update the image with processed result
           image.setImage(result_array)

           # Add metadata to the image (will be collected in _postRun)
           bruise_pct_val = FloatValue("bruise_percentage", "bruise_pct", "Bruise percentage")
           bruise_pct_val.setValue(bruise_pct)
           image.addValue(bruise_pct_val)

           bruise_count_val = IntValue("bruise_count", "bruise_count", "Number of bruise regions")
           bruise_count_val.setValue(bruise_count)
           image.addValue(bruise_count_val)

           threshold_val = IntValue("threshold_used", "threshold", "L threshold used")
           threshold_val.setValue(l_thresh)
           image.addValue(threshold_val)

           return image

       def _postRun(self, results: List[Image]) -> List[Image]:
           """
           Post-processing after all images are processed.

           Called once with all processed Image objects.
           Use this to save images, generate CSV reports, print summaries.

           Args:
               results: List of processed Image objects from _processImage()

           Returns:
               List[Image]: The final list of result images
           """
           print(f"\nSaving {len(results)} images to: {self.output_dir}")

           # Ensure output directory exists
           os.makedirs(self.output_dir, exist_ok=True)

           # Save each image and collect CSV data
           image_io = RGBImageFile()
           csv_data = []

           for image in results:
               # Save the image
               image.saveImage(image_io, self.output_dir)

               # Collect data for CSV
               metadata = image.getMetaData()
               csv_data.append({
                   "filename": image.getImageName(),
                   "bruise_percentage": f"{metadata['bruise_percentage'].getValue():.2f}",
                   "bruise_count": metadata['bruise_count'].getValue(),
                   "lightness_threshold": metadata['threshold_used'].getValue()
               })

               print(f"  Saved: {image.getImageName()} - "
                     f"Bruise: {metadata['bruise_percentage'].getValue():.2f}%")

           # Save CSV results
           self._save_csv(csv_data, self.results_dir)

           print(f"\n{'='*60}")
           print(f"Analysis complete! Processed {len(results)} images.")
           print(f"{'='*60}\n")

           return results

       # =========================================================================
       # HELPER METHODS
       # =========================================================================

       def _detect_bruises(
           self,
           img: NDArray,
           l_thresh: int,
           min_area: int,
           kernel_size: int,
           alpha: float,
           font_scale: float
       ) -> Tuple[NDArray, float, int]:
           """
           Detect bruises on a single fruit image.

           Algorithm:
           1. Convert image to LAB color space
           2. Threshold L channel to find dark regions
           3. Apply morphological operations to remove noise
           4. Filter small regions
           5. Calculate bruise percentage
           6. Annotate image with results

           Args:
               img: Input image as NumPy array (BGR format)
               l_thresh: Lightness threshold
               min_area: Minimum bruise area in pixels
               kernel_size: Morphological kernel size
               alpha: Mask transparency
               font_scale: Text font scale

           Returns:
               Tuple of (annotated_image, bruise_percentage, bruise_count)
           """
           # Convert to LAB color space
           lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
           l_channel, a_channel, b_channel = cv2.split(lab)

           # Threshold L channel - dark regions are potential bruises
           _, binary = cv2.threshold(l_channel, l_thresh, 255, cv2.THRESH_BINARY_INV)

           # Morphological operations to remove noise
           kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
           binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
           binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

           # Find connected components (bruise regions)
           num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
               binary, connectivity=8
           )

           # Filter small regions and count valid bruises
           bruise_mask = np.zeros_like(binary)
           bruise_count = 0
           total_bruise_area = 0

           for i in range(1, num_labels):  # Skip background (label 0)
               area = stats[i, cv2.CC_STAT_AREA]
               if area >= min_area:
                   bruise_mask[labels == i] = 255
                   bruise_count += 1
                   total_bruise_area += area

           # Calculate bruise percentage
           total_pixels = img.shape[0] * img.shape[1]
           bruise_pct = (total_bruise_area / total_pixels) * 100

           # Create visualization
           annotated = self._create_visualization(
               img, bruise_mask, bruise_pct, bruise_count, alpha, font_scale
           )

           return annotated, bruise_pct, bruise_count

       def _create_visualization(
           self,
           img: NDArray,
           bruise_mask: NDArray,
           bruise_pct: float,
           bruise_count: int,
           alpha: float,
           font_scale: float
       ) -> NDArray:
           """
           Create annotated visualization of bruise detection results.

           Args:
               img: Original image
               bruise_mask: Binary mask of bruise regions
               bruise_pct: Bruise percentage
               bruise_count: Number of bruise regions
               alpha: Transparency for mask overlay
               font_scale: Font scale for text

           Returns:
               Annotated image
           """
           # Create colored mask
           mask_color = np.zeros_like(img)
           mask_color[bruise_mask == 255] = self.bruise_color

           # Blend with original image
           result = cv2.addWeighted(img, 1.0, mask_color, alpha, 0)

           # Add text annotations
           thickness = 2
           font = cv2.FONT_HERSHEY_SIMPLEX

           # Bruise percentage
           text1 = f"Bruise: {bruise_pct:.2f}%"
           cv2.putText(result, text1, (20, 50), font, font_scale,
                      (255, 255, 255), thickness, cv2.LINE_AA)

           # Bruise count
           text2 = f"Regions: {bruise_count}"
           cv2.putText(result, text2, (20, 100), font, font_scale,
                      (255, 255, 255), thickness, cv2.LINE_AA)

           return result

       def _save_csv(self, data: List[dict], output_dir: str) -> None:
           """
           Save analysis results to CSV file.

           Args:
               data: List of dictionaries containing results for each image
               output_dir: Directory to save CSV file
           """
           if not data:
               print("Warning: No data to save to CSV.")
               return

           # Ensure directory exists
           os.makedirs(output_dir, exist_ok=True)

           # Create CSV filename
           csv_path = os.path.join(output_dir, f"{self.__analysis_name__}_results.csv")

           # Write CSV
           with open(csv_path, 'w', newline='') as f:
               writer = csv.DictWriter(f, fieldnames=data[0].keys())
               writer.writeheader()
               writer.writerows(data)

           print(f"Results saved to: {csv_path}")


Integration Steps
-----------------

After creating the file, follow these steps to integrate it into Granny:

1. **Update CLI Interface**

   Edit ``Granny/Interfaces/UI/GrannyCLI.py``:

   Add import:

   .. code-block:: python

      from Granny.Analyses.BruiseDetection import BruiseDetection

   Update choices:

   .. code-block:: python

      choices=["segmentation", "blush", "color", "scald", "starch", "bruise"]

2. **Create Test File**

   Create ``tests/test_Analyses/test_BruiseDetection.py``:

   .. code-block:: python

      from Granny.Analyses.BruiseDetection import BruiseDetection
      import pytest


      def test_init():
          """Test analysis initialization."""
          analysis = BruiseDetection()
          assert analysis.__analysis_name__ == "bruise"
          assert analysis.lightness_threshold.getValue() == 80
          assert analysis.min_bruise_area.getValue() == 100


      def test_parameters():
          """Test parameter constraints."""
          analysis = BruiseDetection()

          # Test threshold bounds
          assert analysis.lightness_threshold.getMin() == 0
          assert analysis.lightness_threshold.getMax() == 255

          # Test mask alpha bounds
          assert analysis.mask_alpha.getMin() == 0.0
          assert analysis.mask_alpha.getMax() == 1.0


      def test_parameter_setting():
          """Test setting parameters."""
          analysis = BruiseDetection()

          analysis.lightness_threshold.setValue(90)
          assert analysis.lightness_threshold.getValue() == 90

          analysis.min_bruise_area.setValue(200)
          assert analysis.min_bruise_area.getValue() == 200

Usage Examples
--------------

**Basic usage with defaults:**

.. code-block:: bash

   granny -i cli --analysis bruise --input ./demo/apple_images/

**Custom threshold:**

.. code-block:: bash

   granny -i cli --analysis bruise \
       --input ./images/ \
       --lightness_threshold 75 \
       --min_bruise_area 150

**Full customization:**

.. code-block:: bash

   granny -i cli --analysis bruise \
       --input ./fruit_photos/ \
       --output ./my_results/ \
       --lightness_threshold 70 \
       --min_bruise_area 200 \
       --morph_kernel 7 \
       --mask_alpha 0.7 \
       --font_scale 1.5

**Specify CPU cores:**

.. code-block:: bash

   granny -i cli --analysis bruise \
       --input ./images/ \
       --cpu 4

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   python -m pytest tests/test_Analyses/test_BruiseDetection.py -v

   # Run specific test
   python -m pytest tests/test_Analyses/test_BruiseDetection.py::test_init -v

Key Takeaways
-------------

This example demonstrates:

1. **Proper class structure** with inheritance from ``Analysis``
2. **Three abstract methods**: ``_preRun()``, ``_processImage()``, ``_postRun()``
3. **Parallel processing** via the base class (no manual Pool management)
4. **Type-safe parameters** using IntValue, FloatValue, ImageListValue
5. **Parameter constraints** with min/max ranges
6. **Comprehensive docstrings** for class and methods
7. **Image processing workflow** using OpenCV and NumPy
8. **Result storage** for both images and CSV data
9. **Metadata handling** via ``addValue()`` on Image objects
10. **User feedback** via print statements

Architecture Notes
------------------

**Why three methods instead of one performAnalysis()?**

The base ``Analysis.performAnalysis()`` handles:

- Loading images from the input directory
- Managing the multiprocessing Pool
- CPU core allocation (80% default, or user-specified)
- Calling your methods in the right order

This means you get parallel processing for free just by implementing the three methods correctly.

**Multiprocessing constraints in _processImage():**

Since ``_processImage()`` runs in separate processes:

- Don't modify instance variables (changes won't propagate)
- Return results via the Image object's metadata
- Aggregation happens in ``_postRun()``
- Parameter values can be read (they're pickled with the object)

**Best Practices Demonstrated:**

- **Separation of concerns:** Detection logic separate from visualization
- **Configurable parameters:** All magic numbers are parameters
- **Error handling:** Check for empty image lists
- **Progress feedback:** Print statements for user awareness
- **Consistent naming:** Follow existing Granny conventions
- **Documentation:** Complete docstrings and inline comments
- **Type hints:** Use NDArray and type annotations
- **Timestamped results:** Avoid overwriting previous results
- **CSV output:** Provide machine-readable results

Next Steps
----------

- Modify this example for your specific use case
- Add additional parameters as needed
- Implement more sophisticated algorithms
- Create unit tests for your specific analysis
- Update documentation with your analysis details
