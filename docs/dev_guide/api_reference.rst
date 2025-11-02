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
      :abstractmethod:

      **ABSTRACT:** Perform the analysis on input images.

      This method must be implemented by all subclasses.

      :return: List of processed Image objects
      :rtype: List[Image]

      **Implementation guidelines:**

      1. Load images from ``input_images`` parameter
      2. Process each image
      3. Create result Image objects with results
      4. Add metadata to result images
      5. Save results (images and CSV)
      6. Return list of result images

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

   .. py:method:: addProgramArgs() -> None
      :abstractmethod:

      **ABSTRACT:** Add interface-specific arguments to the argument parser.

      This method is called during initialization to set up command-line arguments
      that the interface needs.

      Example::

         def addProgramArgs(self):
             grp = self.parser.add_argument_group("My Interface Args")
             grp.add_argument("--my-option", type=str, help="...")

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

   .. py:attribute:: required
      :type: bool

      Whether this parameter is required.

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

.. py:class:: ImageListValue(Value)

   Image list/directory parameter type. Represents a directory containing images.

   **Location:** ``Granny/Models/Values/ImageListValue.py``

   **Example:**

   .. code-block:: python

      input_images = ImageListValue(
          "input",
          "input",
          "Directory containing input images"
      )
      input_images.setIsRequired(True)
      input_images.setValue("./demo/images")

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

Image
~~~~~

.. py:class:: Image

   Base class for images in Granny.

   **Location:** ``Granny/Models/Images/Image.py``

   **Methods:**

   .. py:method:: setImageFile(image_file: ImageFile) -> None

      Set the ImageFile object containing the actual image data.

   .. py:method:: getImageFile() -> ImageFile

      Get the ImageFile object.

   .. py:method:: getFileName() -> str

      Get the image filename.

   .. py:method:: addMetadata(metadata: List[dict]) -> None

      Add metadata to the image.

      :param metadata: List of metadata dictionaries with "name" and "value" keys

   .. py:method:: getMetadata() -> List[dict]

      Get all metadata attached to this image.

RGBImage
~~~~~~~~

.. py:class:: RGBImage(Image)

   RGB color image type.

   **Location:** ``Granny/Models/Images/RGBImage.py``

   This is the most common image type used in Granny analyses.

   **Example:**

   .. code-block:: python

      result_image = RGBImage()
      result_file = RGBImageFile()
      result_file.setImage(processed_array)
      result_file.setFileName("result.jpg")
      result_file.setFilePath("./output")
      result_image.setImageFile(result_file)
      result_image.addMetadata([{"name": "score", "value": 95.5}])

ImageFile Classes
-----------------

ImageFile
~~~~~~~~~

.. py:class:: ImageFile

   Base class for image file handlers.

   **Location:** ``Granny/Models/IO/ImageIO.py``

   **Methods:**

   .. py:method:: setImage(image: NDArray) -> None

      Set the image data as a NumPy array.

   .. py:method:: getImage() -> NDArray

      Get the image data as a NumPy array.

   .. py:method:: setFileName(name: str) -> None

      Set the filename.

   .. py:method:: getFileName() -> str

      Get the filename.

   .. py:method:: setFilePath(path: str) -> None

      Set the directory path.

   .. py:method:: getFilePath() -> str

      Get the directory path.

RGBImageFile
~~~~~~~~~~~~

.. py:class:: RGBImageFile(ImageFile)

   RGB image file handler. Uses OpenCV for loading/saving.

   **Location:** ``Granny/Models/IO/RGBImageFile.py``

   Automatically handles image I/O in BGR format (OpenCV convention).

I/O Classes
-----------

ImageIO
~~~~~~~

.. py:class:: ImageIO

   Handles loading and saving of images.

   **Location:** ``Granny/Models/IO/ImageIO.py``

   **Methods:**

   .. py:method:: load(path: str, file_class: Type[ImageFile]) -> List[Image]

      Load all images from a directory.

      :param path: Directory path
      :param file_class: ImageFile class to use (e.g., RGBImageFile)
      :return: List of loaded Image objects

      **Example:**

      .. code-block:: python

         imageIO = ImageIO()
         images = imageIO.load("./input", RGBImageFile)

   .. py:method:: save(images: List[Image]) -> None

      Save all images to their designated paths.

      :param images: List of Image objects to save

      **Example:**

      .. code-block:: python

         imageIO = ImageIO()
         imageIO.save(result_images)

Scheduler Class
---------------

Scheduler
~~~~~~~~~

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

   .. py:method:: schedule() -> List[Any]

      Execute all analyses in dependency order.

      :return: List of results from all analyses

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

   from Granny.Models.IO.ImageIO import ImageIO
   from Granny.Models.IO.RGBImageFile import RGBImageFile

   # Load images
   imageIO = ImageIO()
   images = imageIO.load("./input", RGBImageFile)

   # Process each image
   for image in images:
       img_array = image.getImageFile().getImage()  # Get NumPy array
       # Process img_array with OpenCV/NumPy
       result_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

Creating Result Images
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from Granny.Models.Images.RGBImage import RGBImage
   from Granny.Models.IO.RGBImageFile import RGBImageFile

   # Create result image
   result_image = RGBImage()
   result_file = RGBImageFile()
   result_file.setImage(processed_array)
   result_file.setFileName("output.jpg")
   result_file.setFilePath("./results")
   result_image.setImageFile(result_file)

   # Add metadata
   result_image.addMetadata([
       {"name": "score", "value": 85.5},
       {"name": "threshold", "value": 128}
   ])
   result_image.addMetadata(self.metadata)  # Add standard metadata

   # Save
   imageIO.save([result_image])

Setting Analysis Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

   def performAnalysis(self) -> List[Image]:
       ...

   def getInParams(self) -> Dict[str, Value]:
       ...

   def process_image(self, img: NDArray) -> NDArray:
       ...

See Also
--------

- :doc:`adding_analysis` - Step-by-step guide for creating analyses
- :doc:`adding_interface` - Step-by-step guide for creating interfaces
- :doc:`example_analysis` - Complete working example
