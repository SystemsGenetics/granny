Adding a New Analysis Module
=============================

This guide provides step-by-step instructions for creating a new analysis module in Granny. Analysis modules are pluggable algorithms that process fruit images and generate quality ratings or measurements.

Overview
--------

An analysis module in Granny:

- Inherits from the ``Analysis`` abstract base class
- Defines input parameters using the Value system
- Implements three abstract methods: ``_preRun()``, ``_processImage()``, and ``_postRun()``
- Returns a list of processed ``Image`` objects
- Automatically integrates with all Granny interfaces (CLI, GUI)
- Leverages built-in multiprocessing for parallel image processing
- Can be chained with other analyses using the Scheduler

The Analysis Architecture
-------------------------

The base ``Analysis`` class provides a ``performAnalysis()`` method that:

1. Loads images from the input directory via ``ImageListValue``
2. Calls your ``_preRun()`` method for setup
3. Processes images in parallel using ``multiprocessing.Pool``
4. Calls your ``_postRun()`` method for post-processing and saving results

You implement the three abstract methods to customize behavior:

- ``_preRun()``: Setup before processing (initialize variables, load models, etc.)
- ``_processImage(image)``: Process a single image (runs in parallel across CPU cores)
- ``_postRun(results)``: Post-processing after all images are done (save CSV, cleanup)

The Value System
----------------

Granny uses a type-safe Value system for parameters. Each parameter is represented by a Value object that provides:

- Type checking and validation
- Automatic CLI argument generation
- Default values and required/optional flags
- Help text for users
- Min/max constraints for numeric values

**Available Value Types:**

- ``IntValue`` - Integer parameters (e.g., thresholds, kernel sizes)
- ``FloatValue`` - Floating-point parameters (e.g., confidence scores, alpha values)
- ``StringValue`` - String parameters (e.g., model names, labels)
- ``BoolValue`` - Boolean flags
- ``FileNameValue`` - File paths
- ``FileDirValue`` - Directory paths
- ``ImageListValue`` - Directory containing images (handles loading/saving)
- ``MetaDataValue`` - Metadata storage for results

Step-by-Step Guide
-------------------

Step 1: Create Your Analysis File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new Python file in ``Granny/Analyses/`` with a descriptive name:

.. code-block:: bash

   touch Granny/Analyses/MyNewAnalysis.py

Step 2: Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start your file with necessary imports:

.. code-block:: python

   """
   Brief description of what this analysis does.

   Author: Your Name
   Date: YYYY-MM-DD
   """

   import os
   from datetime import datetime
   from typing import Dict, List, Tuple

   import cv2
   import numpy as np
   from numpy.typing import NDArray

   from Granny.Analyses.Analysis import Analysis
   from Granny.Models.Images.Image import Image
   from Granny.Models.Images.RGBImage import RGBImage
   from Granny.Models.IO.RGBImageFile import RGBImageFile
   from Granny.Models.Values.IntValue import IntValue
   from Granny.Models.Values.FloatValue import FloatValue
   from Granny.Models.Values.StringValue import StringValue
   from Granny.Models.Values.ImageListValue import ImageListValue
   from Granny.Models.Values.MetaDataValue import MetaDataValue

Step 3: Define Your Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create your class inheriting from ``Analysis``:

.. code-block:: python

   class MyNewAnalysis(Analysis):
       """
       Detailed description of your analysis.

       This analysis processes fruit images to [describe what it does].

       Attributes:
           images (List[Image]): List of loaded images for processing
           input_images (ImageListValue): Input directory parameter
           output_images (ImageListValue): Output directory parameter
           my_threshold (IntValue): Example threshold parameter
       """

       __analysis_name__ = "myanalysis"  # Used in CLI: --analysis myanalysis

Step 4: Implement the Constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize your analysis with parameters:

.. code-block:: python

   def __init__(self):
       """
       Initialize the analysis with default parameters.
       """
       super().__init__()

       self.images: List[Image] = []
       self.results_data: List[dict] = []  # For collecting CSV data

       # Required: Input images parameter
       self.input_images = ImageListValue(
           "input",                                    # Machine-readable name
           "input",                                    # CLI argument name (--input)
           "The directory where input images are located."  # Help text
       )
       self.input_images.setIsRequired(True)          # Make it required
       self.addInParam(self.input_images)             # Register as input parameter

       # Output directory for analyzed images
       self.output_images = ImageListValue(
           "output",
           "output",
           "The output directory where analyzed images are written."
       )
       result_dir = os.path.join(
           os.curdir,
           "results",
           self.__analysis_name__,
           datetime.now().strftime("%Y-%m-%d-%H-%M")
       )
       self.output_images.setValue(result_dir)        # Set default value
       self.addInParam(self.output_images)

       # Analysis parameter: threshold
       self.my_threshold = IntValue(
           "threshold",                                # Machine-readable name
           "threshold",                                # CLI argument (--threshold)
           "Threshold value for detection. Range 0-255, default 128."
       )
       self.my_threshold.setMin(0)                    # Set constraints
       self.my_threshold.setMax(255)
       self.my_threshold.setValue(128)                # Set default
       self.my_threshold.setIsRequired(False)         # Make it optional
       self.addInParam(self.my_threshold)             # Register parameter

       # Visualization parameter: mask alpha
       self.mask_alpha = FloatValue(
           "alpha",
           "mask_alpha",
           "Transparency of mask overlay (0.0-1.0, default 0.5)."
       )
       self.mask_alpha.setMin(0.0)
       self.mask_alpha.setMax(1.0)
       self.mask_alpha.setValue(0.5)
       self.mask_alpha.setIsRequired(False)
       self.addInParam(self.mask_alpha)

**Important Notes:**

- The first argument (``name``) is the machine-readable identifier
- The second argument (``label``) becomes the CLI argument: ``--label``
- Always call ``super().__init__()`` first to initialize the base class
- Use ``addInParam()`` for parameters that users can set
- Use ``addRetValue()`` for values that other analyses can use
- Set ``setIsRequired(True)`` for mandatory parameters

Step 5: Implement the Three Abstract Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of overriding ``performAnalysis()``, implement these three methods:

.. code-block:: python

   def _preRun(self):
       """
       Setup before image processing begins.

       This method is called once before any images are processed.
       Use it to:
       - Initialize result containers
       - Load models or resources
       - Print analysis parameters
       """
       self.results_data = []  # Reset results for this run

       # Get output directory
       self.output_dir = self.output_images.getValue()

       # Print analysis info
       print(f"\n{'='*60}")
       print(f"MY NEW ANALYSIS")
       print(f"{'='*60}")
       print(f"Input directory:  {self.input_images.getValue()}")
       print(f"Output directory: {self.output_dir}")
       print(f"Threshold: {self.my_threshold.getValue()}")
       print(f"Mask alpha: {self.mask_alpha.getValue()}")
       print(f"Processing {len(self.images)} images...")
       print(f"{'='*60}\n")

   def _processImage(self, image: Image) -> Image:
       """
       Process a single image.

       This method runs in parallel across multiple CPU cores.
       Each call receives one Image instance and should return a processed Image.

       Args:
           image: The input Image instance to process

       Returns:
           Image: The processed Image with results
       """
       # Get parameter values
       threshold = self.my_threshold.getValue()
       alpha = self.mask_alpha.getValue()

       # Load the image data
       image_io = RGBImageFile()
       image_io.setFilePath(image.getFilePath())
       image.loadImage(image_io)

       # Get the numpy array (BGR format from OpenCV)
       img_array = image.getImage()

       # Perform your analysis
       result_array, metric_value = self._analyze_image(img_array, threshold, alpha)

       # Update the image with the result
       image.setImage(result_array)

       # Add metadata to the image
       metric_val = StringValue("metric", "metric", "Analysis metric value")
       metric_val.setValue(str(metric_value))
       image.addValue(metric_val)

       return image

   def _postRun(self, results: List[Image]) -> List[Image]:
       """
       Post-processing after all images are processed.

       This method is called once after all images have been processed.
       Use it to:
       - Save images to disk
       - Generate CSV reports
       - Print summary statistics

       Args:
           results: List of processed Image objects from _processImage()

       Returns:
           List[Image]: The final list of result images
       """
       print(f"\nSaving {len(results)} images to: {self.output_dir}")

       # Save each image
       image_io = RGBImageFile()
       for image in results:
           image.saveImage(image_io, self.output_dir)

       # Collect data for CSV
       csv_data = []
       for image in results:
           metadata = image.getMetaData()
           csv_data.append({
               "filename": image.getImageName(),
               "metric": metadata.get("metric", StringValue("", "", "")).getValue()
           })

       # Save CSV
       self._save_csv(csv_data, self.output_dir)

       print(f"\n{'='*60}")
       print(f"Analysis complete! Processed {len(results)} images.")
       print(f"{'='*60}\n")

       return results

   def _analyze_image(self, img: NDArray, threshold: int, alpha: float) -> Tuple[NDArray, float]:
       """
       Perform analysis on a single image array.

       Args:
           img: Input image as numpy array (BGR format)
           threshold: Detection threshold
           alpha: Mask transparency

       Returns:
           Tuple of (processed_image, metric_value)
       """
       # Convert to grayscale
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

       # Apply threshold
       _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

       # Calculate metric (e.g., percentage of pixels above threshold)
       metric = (np.sum(binary == 255) / binary.size) * 100

       # Create visualization
       mask_color = np.zeros_like(img)
       mask_color[binary == 255] = [0, 255, 0]  # Green overlay

       # Blend with original
       result = cv2.addWeighted(img, 1.0, mask_color, alpha, 0)

       # Add text annotation
       text = f"Metric: {metric:.2f}%"
       cv2.putText(result, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv2.LINE_AA)

       return result, metric

   def _save_csv(self, data: List[dict], output_dir: str):
       """
       Save analysis results to CSV file.

       Args:
           data: List of dictionaries containing results
           output_dir: Directory to save CSV
       """
       import csv

       os.makedirs(output_dir, exist_ok=True)
       csv_path = os.path.join(output_dir, f"{self.__analysis_name__}_results.csv")

       if not data:
           return

       with open(csv_path, 'w', newline='') as f:
           writer = csv.DictWriter(f, fieldnames=data[0].keys())
           writer.writeheader()
           writer.writerows(data)

       print(f"Results saved to: {csv_path}")

Step 6: Update the CLI Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``Granny/Interfaces/UI/GrannyCLI.py`` to add your analysis to the choices list:

.. code-block:: python

   iface_grp.add_argument(
       "--analysis",
       dest="analysis",
       type=str,
       required=True,
       choices=["segmentation", "blush", "color", "scald", "starch", "myanalysis"],  # Add here
       help="Indicates the analysis to run.",
   )

Also import your analysis class at the top of the file:

.. code-block:: python

   from Granny.Analyses.MyNewAnalysis import MyNewAnalysis

Step 7: Create Tests
~~~~~~~~~~~~~~~~~~~~~

Create a test file ``tests/test_Analyses/test_MyNewAnalysis.py``:

.. code-block:: python

   from Granny.Analyses.MyNewAnalysis import MyNewAnalysis

   def test_init():
       """Test that the analysis initializes correctly."""
       analysis = MyNewAnalysis()
       assert analysis.__analysis_name__ == "myanalysis"
       assert analysis.my_threshold.getValue() == 128

   def test_parameters():
       """Test parameter constraints."""
       analysis = MyNewAnalysis()

       # Test threshold bounds
       assert analysis.my_threshold.getMin() == 0
       assert analysis.my_threshold.getMax() == 255

       # Test alpha bounds
       assert analysis.mask_alpha.getMin() == 0.0
       assert analysis.mask_alpha.getMax() == 1.0

Step 8: Test Your Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run your analysis from the command line:

.. code-block:: bash

   granny -i cli --analysis myanalysis --input ./demo/images/ --threshold 150 --mask_alpha 0.7

Run the test suite:

.. code-block:: bash

   python -m pytest tests/test_Analyses/test_MyNewAnalysis.py -v

Best Practices
--------------

Multiprocessing Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since ``_processImage()`` runs in parallel:

- Don't modify shared state (use return values instead)
- Each image should be processed independently
- Heavy initialization belongs in ``_preRun()``
- Aggregation and saving belongs in ``_postRun()``

Parameter Naming
~~~~~~~~~~~~~~~~

- Use clear, descriptive names for parameters
- Distinguish between analysis parameters (affect results) and visualization parameters (affect output only)
- Use consistent naming across similar parameters (e.g., ``blur_kernel``, ``morph_kernel``)

Documentation
~~~~~~~~~~~~~

- Provide comprehensive docstrings for the class and all methods
- Document parameter ranges, defaults, and units
- Include examples in the module docstring
- Explain the scientific/algorithmic basis of your analysis

Error Handling
~~~~~~~~~~~~~~

- Validate input parameters in ``_preRun()``
- Handle cases where no images are found
- Provide clear error messages to users
- Gracefully handle edge cases (empty images, invalid formats)

Result Storage
~~~~~~~~~~~~~~

- Always save both visual results (images) and numerical results (CSV)
- Include metadata in output images (timestamps, parameters, analysis ID)
- Use consistent directory structures for results
- Generate timestamped result directories to avoid overwriting

Advanced Topics
---------------

Chaining Analyses
~~~~~~~~~~~~~~~~~

Your analysis can depend on outputs from other analyses using the compatibility system:

.. code-block:: python

   def __init__(self):
       super().__init__()

       # Define compatibility with segmentation analysis
       self.compatibility = {
           "segmentation": {
               "input": "output"  # Map our input to segmentation's output
           }
       }

This allows the Scheduler to automatically chain analyses together.

Return Values for Other Analyses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your analysis produces values that other analyses might use:

.. code-block:: python

   # In __init__
   self.fruit_coordinates = MetaDataValue(
       "coordinates",
       "coordinates",
       "Bounding box coordinates of detected fruit"
   )
   self.addRetValue(self.fruit_coordinates)

   # In _postRun()
   self.fruit_coordinates.setValue([(x1, y1, x2, y2), ...])

CPU Core Configuration
~~~~~~~~~~~~~~~~~~~~~~

The base ``Analysis`` class automatically handles CPU core allocation:

- Default (0): Uses 80% of available cores
- User can override via ``--cpu N`` CLI argument
- Set ``self.cpu.setValue(4)`` in ``__init__`` to change default

Working with Image Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add metadata to images using ``Value`` objects:

.. code-block:: python

   from Granny.Models.Values.StringValue import StringValue
   from Granny.Models.Values.FloatValue import FloatValue

   # In _processImage()
   score_val = FloatValue("score", "score", "Analysis score")
   score_val.setValue(95.5)
   image.addValue(score_val)

   # In _postRun(), retrieve metadata
   for image in results:
       metadata = image.getMetaData()
       score = metadata.get("score").getValue()

Troubleshooting
---------------

**Analysis not appearing in CLI:**

- Verify ``__analysis_name__`` is set correctly
- Check that you added the analysis to GrannyCLI's choices list

**Parameters not showing in help:**

- Verify you called ``addInParam()`` for each parameter
- Check that the parameter's ``label`` doesn't conflict with existing parameters
- Ensure you're setting parameters before calling ``addInParam()``

**Images not loading:**

- Verify the input directory path is correct
- Check that images are in a supported format (JPG, PNG, JPEG, TIFF)
- Ensure ``RGBImageFile`` is used correctly with ``setFilePath()``

**Multiprocessing errors:**

- Ensure ``_processImage()`` doesn't access shared mutable state
- Check that all objects passed between processes are picklable
- Move file I/O to ``_preRun()`` or ``_postRun()`` if needed

Next Steps
----------

- Review :doc:`example_analysis` for a complete working example
- Study existing analyses in ``Granny/Analyses/`` for patterns
- Refer to :doc:`api_reference` for detailed API documentation
- Consider contributing your analysis back to the project
