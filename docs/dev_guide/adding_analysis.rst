Adding a New Analysis Module
=============================

This guide provides step-by-step instructions for creating a new analysis module in Granny. Analysis modules are pluggable algorithms that process fruit images and generate quality ratings or measurements.

Overview
--------

An analysis module in Granny:

- Inherits from the ``Analysis`` abstract base class
- Defines input parameters using the Value system
- Implements the ``performAnalysis()`` method
- Returns a list of processed ``Image`` objects
- Automatically integrates with all Granny interfaces (CLI, GUI)
- Can be chained with other analyses using the Scheduler

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
- ``ImageListValue`` - Lists of images (typically input/output directories)
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
   from typing import List

   import cv2
   import numpy as np
   from numpy.typing import NDArray

   from Granny.Analyses.Analysis import Analysis
   from Granny.Models.Images.Image import Image
   from Granny.Models.Images.RGBImage import RGBImage
   from Granny.Models.IO.ImageIO import ImageIO
   from Granny.Models.IO.RGBImageFile import RGBImageFile
   from Granny.Models.Values.IntValue import IntValue
   from Granny.Models.Values.FloatValue import FloatValue
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
           output_results (MetaDataValue): Results directory parameter
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

       # Output directory for CSV results
       self.output_results = MetaDataValue(
           "results",
           "results",
           "The output directory where analysis results are written."
       )
       self.output_results.setValue(result_dir)

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

Step 5: Implement performAnalysis()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the core method that executes your analysis:

.. code-block:: python

   def performAnalysis(self) -> List[Image]:
       """
       Perform the analysis on all input images.

       Returns:
           List[Image]: List of processed Image objects with analysis results
       """
       # Step 1: Load input images
       input_dir = self.input_images.getValue()
       output_dir = self.output_images.getValue()
       results_dir = self.output_results.getValue()

       print(f"Loading images from: {input_dir}")
       imageIO = ImageIO()
       self.images = imageIO.load(input_dir, RGBImageFile)

       if not self.images:
           print("No images found to analyze.")
           return []

       print(f"Processing {len(self.images)} images...")

       # Step 2: Process each image
       result_images = []
       results_data = []  # For CSV output

       for idx, image in enumerate(self.images, 1):
           print(f"  Processing image {idx}/{len(self.images)}: {image.getFileName()}")

           # Get the numpy array
           img_array = image.getImageFile().getImage()

           # Perform your analysis
           result_array, metric_value = self._analyze_image(img_array)

           # Create result image
           result_image = RGBImage()
           result_file = RGBImageFile()
           result_file.setImage(result_array)
           result_file.setFileName(image.getFileName())
           result_file.setFilePath(output_dir)
           result_image.setImageFile(result_file)

           # Add metadata to the image
           result_image.addMetadata(self.metadata)
           result_image.addMetadata([
               {"name": "metric_value", "value": metric_value}
           ])

           result_images.append(result_image)
           results_data.append({
               "filename": image.getFileName(),
               "metric_value": metric_value
           })

       # Step 3: Save results
       print(f"Saving results to: {output_dir}")
       imageIO.save(result_images)

       # Save CSV
       self._save_csv(results_data, results_dir)

       print("Analysis complete!")
       return result_images

   def _analyze_image(self, img: NDArray) -> tuple[NDArray, float]:
       """
       Perform analysis on a single image.

       Args:
           img: Input image as numpy array (BGR format)

       Returns:
           Tuple of (processed_image, metric_value)
       """
       # Get parameter values
       threshold = self.my_threshold.getValue()
       alpha = self.mask_alpha.getValue()

       # Your image processing logic here
       # This is a simple example - replace with your actual algorithm

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

Step 6: Add Your Analysis to the Import Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``Granny/Analyses/__init__.py`` to include your new analysis:

.. code-block:: python

   from .MyNewAnalysis import MyNewAnalysis

This makes your analysis discoverable by the CLI interface.

Step 7: Update the CLI Interface
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

Step 8: Create Tests
~~~~~~~~~~~~~~~~~~~~~

Create a test file ``tests/test_Analyses/test_MyNewAnalysis.py``:

.. code-block:: python

   from Granny.Analyses.MyNewAnalysis import MyNewAnalysis

   def test_init():
       """Test that the analysis initializes correctly."""
       analysis = MyNewAnalysis()
       assert analysis.__analysis_name__ == "myanalysis"
       assert analysis.my_threshold.getValue() == 128

   def test_performAnalysis():
       """Test the analysis with sample data."""
       analysis = MyNewAnalysis()

       # Set test parameters
       analysis.input_images.setValue("test-assets/sample_images")
       analysis.my_threshold.setValue(100)

       # Run analysis
       results = analysis.performAnalysis()

       # Verify results
       assert isinstance(results, list)
       # Add more specific assertions based on your analysis

Step 9: Test Your Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run your analysis from the command line:

.. code-block:: bash

   granny -i cli --analysis myanalysis --input ./demo/images/ --threshold 150 --mask_alpha 0.7

Run the test suite:

.. code-block:: bash

   python -m pytest tests/test_Analyses/test_MyNewAnalysis.py -v

Best Practices
--------------

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

Performance
~~~~~~~~~~~

- Process images efficiently using NumPy operations
- Consider multiprocessing for independent image processing (see ``BlushColor`` for example)
- Minimize memory usage for large image sets
- Provide progress feedback via print statements

Error Handling
~~~~~~~~~~~~~~

- Validate input parameters in ``performAnalysis()``
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

   # In performAnalysis()
   self.fruit_coordinates.setValue([(x1, y1, x2, y2), ...])

Custom Image Types
~~~~~~~~~~~~~~~~~~

If you need a specialized image type beyond ``RGBImage``:

1. Create a new class inheriting from ``Image`` in ``Granny/Models/Images/``
2. Create a corresponding file class inheriting from ``ImageFile`` in ``Granny/Models/IO/``
3. Implement required methods for loading and saving

Troubleshooting
---------------

**Analysis not appearing in CLI:**

- Verify ``__analysis_name__`` is set correctly
- Check that you added the analysis to GrannyCLI's choices list
- Ensure the import statement is in ``Granny/Analyses/__init__.py``

**Parameters not showing in help:**

- Verify you called ``addInParam()`` for each parameter
- Check that the parameter's ``label`` doesn't conflict with existing parameters
- Ensure you're setting parameters before calling ``addInParam()``

**Images not loading:**

- Verify the input directory path is correct
- Check that images are in a supported format (JPG, PNG)
- Ensure ``ImageIO`` is initialized correctly
- Use ``RGBImageFile`` for standard RGB images

**Results not saving:**

- Verify the output directory is writable
- Check that ``imageIO.save()`` is called with the result images
- Ensure each image has both an ImageFile and metadata set
- Verify directory paths are created with ``os.makedirs(path, exist_ok=True)``

Next Steps
----------

- Review :doc:`example_analysis` for a complete working example
- Study existing analyses in ``Granny/Analyses/`` for patterns
- Refer to :doc:`api_reference` for detailed API documentation
- Consider contributing your analysis back to the project
