API Reference
=============

This document provides detailed API documentation for extending Granny with new analysis modules and interfaces.

Core Classes
------------

Analysis Base Class
~~~~~~~~~~~~~~~~~~~

.. py:class:: Analysis

   Abstract base class for all analysis modules.

   All custom analyses must inherit from this class and implement the abstract methods.

   **Location:** ``Granny/Analyses/Analysis.py``

   **Class Attributes:**

   .. py:attribute:: __analysis_name__
      :type: str

      Machine-readable name for the analysis. Used in CLI ``--analysis`` argument.

      Example: ``"segmentation"``, ``"starch"``, ``"blush"``

   **Instance Attributes:**

   .. py:attribute:: in_params
      :type: Dict[str, Value]

      Dictionary of input parameters. Keys are parameter names, values are Value objects.

   .. py:attribute:: ret_values
      :type: Dict[str, Value]

      Dictionary of return values that other analyses can use.

   .. py:attribute:: compatibility
      :type: Dict[str, Dict[str, str]]

      Mapping of compatible analyses. Format::

         {
             "other_analysis_name": {
                 "my_param": "their_return_value"
             }
         }

   .. py:attribute:: metadata
      :type: List[Value]

      List of metadata values automatically added to all analyses:
      - ``dt`` - Analysis date/time
      - ``id`` - Unique analysis identifier (UUID)
      - ``path`` - Current directory path

   .. py:attribute:: cpu
      :type: IntValue

      Number of CPU cores for parallel processing. Default 0 = auto (80% of cores).

   .. py:attribute:: images
      :type: List[Image]

      List of loaded images (populated by ``performAnalysis()``).

   **Methods:**

   .. py:method:: __init__()

      Initialize the analysis. Must call ``super().__init__()`` first.

      Subclasses should:
      - Initialize instance variables
      - Define input parameters using Value objects
      - Set default values for parameters
      - Register parameters with ``addInParam()``

   .. py:method:: addInParam(*params: Value) -> None

      Add one or more input parameters to the analysis.

      :param params: Variable number of Value objects to register
      :type params: Value

      Example::

         threshold = IntValue("th", "threshold", "Detection threshold")
         self.addInParam(threshold)

   .. py:method:: getInParams() -> Dict[str, Value]

      Get all input parameters.

      :return: Dictionary of parameter names to Value objects
      :rtype: Dict[str, Value]

   .. py:method:: resetInParams() -> None

      Clear all input parameters. Used when re-configuring an analysis instance.

   .. py:method:: addRetValue(*values: Value) -> None

      Add one or more return values that other analyses can use.

      :param values: Variable number of Value objects
      :type values: Value

   .. py:method:: getRetValues() -> Dict[str, Value]

      Get all return values.

      :return: Dictionary of return value names to Value objects
      :rtype: Dict[str, Value]

   .. py:method:: resetRetValues() -> None

      Clear all return values.

   .. py:method:: performAnalysis() -> List[Image]

      Perform the analysis on input images. **Do not override this method.**

      The default implementation:
      1. Loads images from ``input_images`` parameter via ``ImageListValue.readValue()``
      2. Calls ``_preRun()`` for setup
      3. Processes images in parallel using ``multiprocessing.Pool``
      4. Calls ``_postRun()`` with results

      :return: List of processed Image objects (from ``_postRun()``)
      :rtype: List[Image]

   **Abstract Methods (must implement):**

   .. py:method:: _preRun() -> None
      :abstractmethod:

      Setup before image processing begins. Called once before any images are processed.

      Use this for:
      - Initializing result containers
      - Loading models or resources
      - Printing analysis parameters

   .. py:method:: _processImage(image: Image) -> Image
      :abstractmethod:

      Process a single image. Runs in parallel across CPU cores.

      :param image: Input Image instance
      :return: Processed Image instance
      :rtype: Image

      **Important:** This method runs in separate processes. Avoid modifying shared state.

   .. py:method:: _postRun(results: List[Image]) -> List[Image]
      :abstractmethod:

      Post-processing after all images are done. Called once with all results.

      Use this for:
      - Saving images to disk
      - Generating CSV reports
      - Computing aggregate statistics

      :param results: List of processed Image objects from ``_processImage()``
      :return: Final list of result images
      :rtype: List[Image]

GrannyUI Base Class
~~~~~~~~~~~~~~~~~~~

.. py:class:: GrannyUI

   Abstract base class for user interfaces.

   **Location:** ``Granny/Interfaces/UI/GrannyUI.py``

   **Constructor:**

   .. py:method:: __init__(parser: ArgumentParser)

      Initialize the interface.

      :param parser: ArgumentParser instance for command-line arguments
      :type parser: argparse.ArgumentParser

   **Instance Attributes:**

   .. py:attribute:: parser
      :type: ArgumentParser

      The argparse ArgumentParser instance used for handling command-line arguments.

   **Abstract Methods:**

   .. py:method:: run() -> None
      :abstractmethod:

      **ABSTRACT:** Execute the interface.

      This method is called to start the interface and must be implemented by all subclasses.

      Typical implementation:
      1. Parse command-line arguments
      2. Get user input (CLI args, GUI forms, web requests, etc.)
      3. Instantiate selected analysis
      4. Set analysis parameters from user input
      5. Call ``analysis.performAnalysis()``
      6. Handle/display results

   **Note:** ``addProgramArgs()`` is commonly implemented but not required by the base class.
   See ``GrannyCLI`` for an example implementation.

Value Classes
-------------

All Value classes inherit from the abstract ``Value`` base class and provide type-safe parameter handling.

**Location:** ``Granny/Models/Values/``

Value Base Class
~~~~~~~~~~~~~~~~

.. py:class:: Value

   Abstract base class for all parameter value types.

   **Location:** ``Granny/Models/Values/Value.py``

   **Constructor:**

   .. py:method:: __init__(name: str, label: str, help: str)

      :param name: Machine-readable parameter name
      :param label: Human-readable label (used for CLI arguments)
      :param help: Help text describing the parameter

   **Attributes:**

   .. py:attribute:: name
      :type: str

      Machine-readable name for the value.

   .. py:attribute:: label
      :type: str

      Human-readable label. Becomes CLI argument: ``--{label}``

   .. py:attribute:: help
      :type: str

      Help text displayed to users.

   .. py:attribute:: type
      :type: Type

      Python type of the value (int, float, str, etc.).

   .. py:attribute:: value
      :type: Any

      The current value.

   .. py:attribute:: is_set
      :type: bool

      Whether the user has set this value.

   **Methods:**

   .. py:method:: validate(value: Any) -> bool
      :abstractmethod:

      Validate that the value meets constraints. Must be implemented by subclasses.

   .. py:method:: getName() -> str

      Get the machine-readable name.

   .. py:method:: getLabel() -> str

      Get the human-readable label (used for CLI args).

   .. py:method:: getHelp() -> str

      Get the help text.

   .. py:method:: getType() -> Type

      Get the Python type of this value.

   .. py:method:: setType(type: Type) -> None

      Set the Python type.

   .. py:method:: setValue(value: Any) -> None

      Set the value. Calls ``validate()`` first.

   .. py:method:: getValue() -> Any

      Get the current value.

   .. py:method:: isSet() -> bool

      Check if the user has set this value.

   .. py:method:: setIsRequired(is_required: bool) -> None

      Set whether this parameter is required.

   .. py:method:: getIsRequired() -> bool

      Check if this parameter is required.

IntValue
~~~~~~~~

.. py:class:: IntValue(Value)

   Integer parameter type with min/max constraints.

   **Location:** ``Granny/Models/Values/IntValue.py``

   **Additional Methods:**

   .. py:method:: setMin(min_val: int) -> None

      Set minimum allowed value.

   .. py:method:: getMin() -> int

      Get minimum allowed value.

   .. py:method:: setMax(max_val: int) -> None

      Set maximum allowed value.

   .. py:method:: getMax() -> int

      Get maximum allowed value.

   **Example:**

   .. code-block:: python

      threshold = IntValue("th", "threshold", "Detection threshold (0-255)")
      threshold.setMin(0)
      threshold.setMax(255)
      threshold.setValue(128)
      threshold.setIsRequired(False)

FloatValue
~~~~~~~~~~

.. py:class:: FloatValue(Value)

   Floating-point parameter type with min/max constraints.

   **Location:** ``Granny/Models/Values/FloatValue.py``

   **Additional Methods:**

   .. py:method:: setMin(min_val: float) -> None

      Set minimum allowed value.

   .. py:method:: getMin() -> float

      Get minimum allowed value.

   .. py:method:: setMax(max_val: float) -> None

      Set maximum allowed value.

   .. py:method:: getMax() -> float

      Get maximum allowed value.

   **Example:**

   .. code-block:: python

      alpha = FloatValue("alpha", "mask_alpha", "Mask transparency (0.0-1.0)")
      alpha.setMin(0.0)
      alpha.setMax(1.0)
      alpha.setValue(0.5)

StringValue
~~~~~~~~~~~

.. py:class:: StringValue(Value)

   String parameter type.

   **Location:** ``Granny/Models/Values/StringValue.py``

   **Example:**

   .. code-block:: python

      model = StringValue("model", "model", "Model name or path")
      model.setValue("pome_fruit-v1_0")

BoolValue
~~~~~~~~~

.. py:class:: BoolValue(Value)

   Boolean flag parameter type.

   **Location:** ``Granny/Models/Values/BoolValue.py``

   **Example:**

   .. code-block:: python

      debug = BoolValue("debug", "debug", "Enable debug mode")
      debug.setValue(False)

FileNameValue
~~~~~~~~~~~~~

.. py:class:: FileNameValue(Value)

   File path parameter type. Validates that path is a file or a valid model name.

   **Location:** ``Granny/Models/Values/FileNameValue.py``

   **Example:**

   .. code-block:: python

      input_file = FileNameValue("input", "input_file", "Path to input file")
      input_file.setValue("/path/to/file.jpg")

FileDirValue
~~~~~~~~~~~~

.. py:class:: FileDirValue(Value)

   Directory path parameter type. Creates directory if it doesn't exist.

   **Location:** ``Granny/Models/Values/FileDirValue.py``

   **Example:**

   .. code-block:: python

      output = FileDirValue("out", "output", "Output directory")
      output.setValue("./results/analysis")  # Directory will be created

ImageListValue
~~~~~~~~~~~~~~

.. py:class:: ImageListValue(FileDirValue)

   Image list/directory parameter type. Represents a directory containing images.
   Handles loading and saving of image lists.

   **Location:** ``Granny/Models/Values/ImageListValue.py``

   **Additional Methods:**

   .. py:method:: readValue() -> None

      Load all images from the directory into the internal image list.
      Called automatically by ``Analysis.performAnalysis()``.

   .. py:method:: writeValue() -> None

      Save all images in the internal list to the directory.

   .. py:method:: getImageList() -> List[Image]

      Get the list of loaded Image objects.

   .. py:method:: setImageList(images: List[Image]) -> None

      Set the list of Image objects.

   **Example:**

   .. code-block:: python

      input_images = ImageListValue(
          "input",
          "input",
          "Directory containing input images"
      )
      input_images.setIsRequired(True)
      input_images.setValue("./demo/images")

      # After readValue() is called:
      images = input_images.getImageList()

MetaDataValue
~~~~~~~~~~~~~

.. py:class:: MetaDataValue(Value)

   Metadata storage parameter type. Used for results directories and metadata.

   **Location:** ``Granny/Models/Values/MetaDataValue.py``

   **Example:**

   .. code-block:: python

      results = MetaDataValue(
          "results",
          "results",
          "Results output directory"
      )
      results.setValue("./results/analysis_2024-01-01")

Image Classes
-------------

Image Base Class
~~~~~~~~~~~~~~~~

.. py:class:: Image

   Abstract base class for images in Granny.

   **Location:** ``Granny/Models/Images/Image.py``

   **Constructor:**

   .. py:method:: __init__(filepath: str)

      Initialize the Image with a file path.

      :param filepath: Path to the image file

   **Attributes:**

   .. py:attribute:: filepath
      :type: str

      Absolute file path of the image.

   .. py:attribute:: image
      :type: NDArray[np.uint8]

      The image data as a NumPy array.

   .. py:attribute:: metadata
      :type: Dict[str, Value]

      Dictionary of metadata values attached to this image.

   .. py:attribute:: results
      :type: Any

      Segmentation results (for use with YOLO models).

   **Methods:**

   .. py:method:: addValue(*values: Value) -> None

      Add metadata values to the image.

      :param values: One or more Value objects to attach

      Example::

         score = FloatValue("score", "score", "Analysis score")
         score.setValue(95.5)
         image.addValue(score)

   .. py:method:: getValue(key: str) -> Value

      Get a metadata value by name.

      :param key: The name of the metadata value
      :return: The Value object

   .. py:method:: getFilePath() -> str

      Get the absolute file path.

   .. py:method:: getImageName() -> str

      Get the filename (without path).

   .. py:method:: getShape() -> tuple

      Get the image dimensions (height, width, channels).

   **Abstract Methods:**

   .. py:method:: loadImage(image_io: ImageIO) -> None
      :abstractmethod:

      Load image data using the provided ImageIO instance.

   .. py:method:: saveImage(image_io: ImageIO, folder: str) -> None
      :abstractmethod:

      Save image data to the specified folder.

   .. py:method:: getImage() -> NDArray[np.uint8]
      :abstractmethod:

      Get the image data as a NumPy array.

   .. py:method:: setImage(image: NDArray[np.uint8]) -> None
      :abstractmethod:

      Set the image data from a NumPy array.

   .. py:method:: getMetaData() -> Dict[str, Value]
      :abstractmethod:

      Get all metadata attached to this image.

   .. py:method:: setMetaData(metadata: Dict[str, Value]) -> None
      :abstractmethod:

      Set the metadata for the image.

RGBImage
~~~~~~~~

.. py:class:: RGBImage(Image)

   RGB color image type. The most common image type used in Granny analyses.

   **Location:** ``Granny/Models/Images/RGBImage.py``

   **Constructor:**

   .. py:method:: __init__(filepath: str)

      Initialize with a file path.

      :param filepath: Path to the image file

   **Additional Methods:**

   .. py:method:: toRGB() -> None

      Convert image from BGR to RGB format.

   .. py:method:: toBGR() -> None

      Convert image from RGB to BGR format.

   .. py:method:: rotateImage() -> None

      Rotate the image 90 degrees clockwise.

   **Example:**

   .. code-block:: python

      # Load an image
      image = RGBImage("/path/to/image.jpg")
      image_io = RGBImageFile()
      image_io.setFilePath(image.getFilePath())
      image.loadImage(image_io)

      # Process the image
      img_array = image.getImage()
      processed = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
      image.setImage(processed)

      # Add metadata
      score = FloatValue("score", "score", "Analysis score")
      score.setValue(95.5)
      image.addValue(score)

      # Save
      image.saveImage(image_io, "./output")

ImageIO Classes
---------------

ImageIO Base Class
~~~~~~~~~~~~~~~~~~

.. py:class:: ImageIO

   Abstract base class for image file handlers. Handles loading and saving individual images.

   **Location:** ``Granny/Models/IO/ImageIO.py``

   **Attributes:**

   .. py:attribute:: filepath
      :type: str

      Full path to the image file.

   .. py:attribute:: image_dir
      :type: str

      Directory containing the image.

   .. py:attribute:: image_name
      :type: str

      Filename of the image.

   **Methods:**

   .. py:method:: setFilePath(filepath: str) -> None

      Set the file path and extract directory/filename.

      :param filepath: Full path to the image file

   **Abstract Methods:**

   .. py:method:: loadImage() -> NDArray[np.uint8]
      :abstractmethod:

      Load and return the image data.

   .. py:method:: saveImage(image: NDArray[np.uint8], output_path: str) -> None
      :abstractmethod:

      Save the image to the specified directory.

   .. py:method:: getType() -> str
      :abstractmethod:

      Get the image type (e.g., "rgb", "gray").

RGBImageFile
~~~~~~~~~~~~

.. py:class:: RGBImageFile(ImageIO)

   RGB image file handler. Uses OpenCV for loading/saving.

   **Location:** ``Granny/Models/IO/RGBImageFile.py``

   Images are loaded/saved in BGR format (OpenCV convention).

   **Example:**

   .. code-block:: python

      from Granny.Models.IO.RGBImageFile import RGBImageFile

      # Load an image
      image_io = RGBImageFile()
      image_io.setFilePath("/path/to/image.jpg")
      img_array = image_io.loadImage()  # Returns NDArray in BGR format

      # Save an image
      image_io.saveImage(img_array, "/output/directory")

Scheduler Class
---------------

.. py:class:: Scheduler

   Manages dependencies between analyses and executes them in correct order.

   **Location:** ``Granny/Interfaces/Scheduler/Scheduler.py``

   Uses a directed acyclic graph (DAG) with topological sorting to handle analysis dependencies.

   **Methods:**

   .. py:method:: __init__()

      Initialize the scheduler.

   .. py:method:: add_analysis(analysis: Analysis, dependencies: List[Analysis]) -> None

      Add an analysis with its dependencies.

      :param analysis: Analysis to add
      :param dependencies: List of Analysis objects this analysis depends on

      **Example:**

      .. code-block:: python

         scheduler = Scheduler()

         segmentation = Segmentation()
         starch = StarchArea()

         scheduler.add_analysis(segmentation, [])
         scheduler.add_analysis(starch, [segmentation])

   .. py:method:: schedule() -> List[int]

      Determine execution order based on dependencies.

      :return: List of analysis IDs in the order they should be run
      :raises ValueError: If there is a cycle in the dependencies

   .. py:method:: run() -> None

      Execute all analyses in dependency order.

      Calls ``schedule()`` internally, then runs each analysis via ``performAnalysis()``.

Utility Functions
-----------------

Analysis Discovery
~~~~~~~~~~~~~~~~~~

To discover all available analyses:

.. code-block:: python

   from Granny.Analyses.Analysis import Analysis

   # Get all analysis classes
   analyses = Analysis.__subclasses__()

   # Get analysis by name
   for cls in analyses:
       if cls.__analysis_name__ == "segmentation":
           analysis = cls()
           break

Parameter Introspection
~~~~~~~~~~~~~~~~~~~~~~~

To inspect parameters of an analysis:

.. code-block:: python

   analysis = StarchArea()
   params = analysis.getInParams()

   for param_name, param_obj in params.items():
       print(f"Parameter: {param_obj.getLabel()}")
       print(f"  Type: {param_obj.getType()}")
       print(f"  Default: {param_obj.getValue()}")
       print(f"  Required: {param_obj.getIsRequired()}")
       print(f"  Help: {param_obj.getHelp()}")

       if hasattr(param_obj, 'getMin'):
           print(f"  Range: {param_obj.getMin()} - {param_obj.getMax()}")

Common Patterns
---------------

Loading and Processing Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from Granny.Models.Images.RGBImage import RGBImage
   from Granny.Models.IO.RGBImageFile import RGBImageFile

   # In _processImage():
   def _processImage(self, image: Image) -> Image:
       # Load image data
       image_io = RGBImageFile()
       image_io.setFilePath(image.getFilePath())
       image.loadImage(image_io)

       # Get NumPy array (BGR format)
       img_array = image.getImage()

       # Process with OpenCV/NumPy
       result = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

       # Update image
       image.setImage(result)
       return image

Saving Results in _postRun()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from Granny.Models.IO.RGBImageFile import RGBImageFile

   def _postRun(self, results: List[Image]) -> List[Image]:
       output_dir = self.output_images.getValue()

       image_io = RGBImageFile()
       for image in results:
           image.saveImage(image_io, output_dir)

       return results

Adding Metadata to Images
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from Granny.Models.Values.StringValue import StringValue
   from Granny.Models.Values.FloatValue import FloatValue

   # Add metadata in _processImage()
   score = FloatValue("score", "score", "Analysis score")
   score.setValue(95.5)
   image.addValue(score)

   # Retrieve metadata in _postRun()
   for image in results:
       metadata = image.getMetaData()
       if "score" in metadata:
           score_value = metadata["score"].getValue()

Setting Analysis Parameters (in interfaces)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get parameters
   params = analysis.getInParams()

   # Clear and reset
   analysis.resetInParams()

   # Set each parameter
   for param in params.values():
       label = param.getLabel()
       if label in user_values:
           param.setValue(user_values[label])
       # else: uses default value

       analysis.addInParam(param)

Type Annotations
----------------

For better type checking, use these type hints:

.. code-block:: python

   from typing import List, Dict, Any, Optional
   from numpy.typing import NDArray

   from Granny.Models.Images.Image import Image
   from Granny.Models.Values.Value import Value

   def _processImage(self, image: Image) -> Image:
       ...

   def _postRun(self, results: List[Image]) -> List[Image]:
       ...

   def getInParams(self) -> Dict[str, Value]:
       ...

   def process_array(self, img: NDArray) -> NDArray:
       ...

See Also
--------

- :doc:`adding_analysis` - Step-by-step guide for creating analyses
- :doc:`adding_interface` - Step-by-step guide for creating interfaces
- :doc:`example_analysis` - Complete working example
