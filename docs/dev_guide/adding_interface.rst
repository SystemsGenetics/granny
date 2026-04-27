Adding a New Interface
======================

This guide provides step-by-step instructions for creating a new user interface for Granny. Interface modules provide different ways for users to interact with Granny's analysis capabilities.

Overview
--------

An interface module in Granny:

- Inherits from the ``GrannyUI`` abstract base class
- Integrates with Python's ``argparse`` for argument handling
- Dynamically discovers and exposes all available analyses
- Automatically generates UI elements based on analysis parameters
- Manages the execution of analyses with user-provided values

Existing interfaces include:

- **GrannyCLI** - Command-line interface for batch processing
- **GrannyPyQt** - Graphical desktop interface (PyQt5)
- **Mobile** - Mobile interface (placeholder for future development)

Use Cases for New Interfaces
-----------------------------

You might want to create a new interface for:

- Web-based interface (Flask/Django)
- REST API for programmatic access
- Jupyter notebook integration
- Desktop GUI with different framework (Tkinter, wxPython)
- Mobile applications (iOS/Android)
- Voice-controlled interface
- Batch processing scripts with configuration files

The Interface Architecture
---------------------------

Interfaces in Granny follow a specific workflow:

1. **Argument Setup Phase:**

   - Interface is instantiated with an ``ArgumentParser``
   - Interface adds its own specific arguments via ``addProgramArgs()``
   - Interface discovers available analyses
   - Interface adds analysis-specific arguments to the parser

2. **Execution Phase:**

   - The ``run()`` method is called
   - Interface parses command-line arguments (or gets user input)
   - Interface instantiates the selected analysis
   - Interface sets analysis parameters from user input
   - Interface calls ``analysis.performAnalysis()``
   - Interface handles the results

Step-by-Step Guide
-------------------

Step 1: Create Your Interface File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new Python file in ``Granny/Interfaces/UI/`` (or appropriate subdirectory):

.. code-block:: bash

   touch Granny/Interfaces/UI/GrannyWeb.py

Step 2: Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start your file with necessary imports:

.. code-block:: python

   """
   Web-based interface for Granny using Flask.

   Author: Your Name
   Date: YYYY-MM-DD
   """

   import argparse
   from typing import Optional

   from Granny.Analyses.Analysis import Analysis
   from Granny.Interfaces.UI.GrannyUI import GrannyUI


   # Import all available analyses
   from Granny.Analyses.Segmentation import Segmentation
   from Granny.Analyses.BlushColor import BlushColor
   from Granny.Analyses.PeelColor import PeelColor
   from Granny.Analyses.StarchArea import StarchArea
   from Granny.Analyses.SuperficialScald import SuperficialScald

Step 3: Define Your Interface Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create your class inheriting from ``GrannyUI``:

.. code-block:: python

   class GrannyWeb(GrannyUI):
       """
       Web-based interface for Granny analysis.

       This interface provides a web server that allows users to
       upload images and run analyses through a browser.

       Attributes:
           analysis_name (str): Name of the selected analysis
           port (int): Port number for the web server
       """

       def __init__(self, parser: argparse.ArgumentParser):
           """
           Initialize the web interface.

           Args:
               parser: ArgumentParser instance for handling command-line arguments
           """
           super().__init__(parser)
           self.analysis_name: str = ""
           self.port: int = 5000

Step 4: Implement addProgramArgs() (Optional but Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method adds interface-specific arguments to the argument parser.

**Note:** ``addProgramArgs()`` is NOT an abstract method in the base class - only ``run()``
is required. However, most interfaces implement this method to add their own arguments:

.. code-block:: python

   def addProgramArgs(self) -> None:
       """
       Add web interface specific arguments to the argument parser.

       This method is called during initialization to set up all
       command-line arguments that this interface needs.
       """
       # Add interface-specific arguments
       iface_grp = self.parser.add_argument_group("Web interface args")

       iface_grp.add_argument(
           "--port",
           dest="port",
           type=int,
           default=5000,
           help="Port number for the web server (default: 5000)",
       )

       iface_grp.add_argument(
           "--host",
           dest="host",
           type=str,
           default="127.0.0.1",
           help="Host address for the web server (default: 127.0.0.1)",
       )

       iface_grp.add_argument(
           "--debug",
           dest="debug",
           action="store_true",
           help="Run server in debug mode",
       )

**Note:** Unlike the CLI interface, a web interface might not need to specify
analysis types in command-line arguments, as users will select analyses through
the web UI. However, you may want to add arguments for server configuration.

Step 5: Implement run()
~~~~~~~~~~~~~~~~~~~~~~~

This is the main entry point for your interface:

.. code-block:: python

   def run(self):
       """
       Start the web interface.

       This method launches the Flask web server and handles
       user interactions through HTTP requests.
       """
       # Parse command-line arguments
       args, _ = self.parser.parse_known_args()

       # Import Flask here to avoid dependency if not using this interface
       try:
           from flask import Flask, render_template, request, jsonify
       except ImportError:
           print("Error: Flask is not installed.")
           print("Install it with: pip install flask")
           return

       # Create Flask app
       app = Flask(__name__)

       # Store reference to self for route handlers
       interface = self

       @app.route('/')
       def index():
           """Serve the main page."""
           # Get list of available analyses
           analyses = Analysis.__subclasses__()
           analysis_info = [
               {
                   "name": cls.__analysis_name__,
                   "class": cls.__name__
               }
               for cls in analyses
           ]
           return render_template('index.html', analyses=analysis_info)

       @app.route('/run_analysis', methods=['POST'])
       def run_analysis():
           """Handle analysis execution requests."""
           try:
               # Get analysis name from request
               analysis_name = request.form.get('analysis')
               if not analysis_name:
                   return jsonify({"error": "No analysis specified"}), 400

               # Find and instantiate the analysis
               analysis = interface._get_analysis(analysis_name)
               if not analysis:
                   return jsonify({"error": f"Unknown analysis: {analysis_name}"}), 400

               # Set parameters from form data
               interface._set_analysis_params(analysis, request.form)

               # Run the analysis
               results = analysis.performAnalysis()

               return jsonify({
                   "success": True,
                   "message": f"Processed {len(results)} images",
                   "results": [img.getImageName() for img in results]
               })

           except Exception as e:
               return jsonify({"error": str(e)}), 500

       # Start the server
       print(f"Starting Granny web interface on {args.host}:{args.port}")
       print(f"Open your browser to: http://{args.host}:{args.port}")
       app.run(host=args.host, port=args.port, debug=args.debug)

   def _get_analysis(self, analysis_name: str) -> Optional[Analysis]:
       """
       Get an analysis instance by name.

       Args:
           analysis_name: Machine-readable name of the analysis

       Returns:
           Analysis instance or None if not found
       """
       analyses = Analysis.__subclasses__()
       for cls in analyses:
           if cls.__analysis_name__ == analysis_name:
               return cls()
       return None

   def _set_analysis_params(self, analysis: Analysis, form_data: dict) -> None:
       """
       Set analysis parameters from form data.

       Args:
           analysis: Analysis instance to configure
           form_data: Dictionary of form parameters
       """
       params = analysis.getInParams()
       analysis.resetInParams()

       for param in params.values():
           # Get value from form data
           form_key = param.getLabel()
           if form_key in form_data:
               value = form_data[form_key]

               # Convert to appropriate type
               param_type = param.getType()
               if param_type == int:
                   value = int(value)
               elif param_type == float:
                   value = float(value)

               param.setValue(value)
               print(f"  {param.getLabel()}: {value}")
           else:
               # Use default value
               print(f"  {param.getLabel()}: (default) {param.getValue()}")

           analysis.addInParam(param)

Step 6: Register Your Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``Granny/GrannyBase.py`` to register your new interface:

.. code-block:: python

   # At the top, import your interface
   from Granny.Interfaces.UI.GrannyWeb import GrannyWeb

   # In the run() function, add your interface to the choices
   parser.add_argument(
       "-i",
       dest="interface",
       type=str,
       required=True,
       choices=["cli", "gui", "web"],  # Add "web" here
       help="Indicates the user interface to use.",
   )

   # Add elif branch for your interface
   elif interface == "web":
       iface = GrannyWeb(parser)

Step 7: Create Required Assets (if applicable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a web interface, you'll need HTML templates:

Create ``templates/index.html``:

.. code-block:: html

   <!DOCTYPE html>
   <html>
   <head>
       <title>Granny - Fruit Quality Analysis</title>
   </head>
   <body>
       <h1>Granny Analysis</h1>

       <form id="analysisForm" action="/run_analysis" method="POST">
           <label for="analysis">Select Analysis:</label>
           <select name="analysis" id="analysis" required>
               {% for analysis in analyses %}
               <option value="{{ analysis.name }}">{{ analysis.class }}</option>
               {% endfor %}
           </select>

           <label for="input">Input Directory:</label>
           <input type="text" name="input" id="input" required>

           <button type="submit">Run Analysis</button>
       </form>

       <div id="results"></div>

       <script>
           document.getElementById('analysisForm').onsubmit = async (e) => {
               e.preventDefault();
               const formData = new FormData(e.target);
               const response = await fetch('/run_analysis', {
                   method: 'POST',
                   body: formData
               });
               const result = await response.json();
               document.getElementById('results').innerHTML =
                   JSON.stringify(result, null, 2);
           };
       </script>
   </body>
   </html>

Example: Alternative CLI Interface
-----------------------------------

Here's an example of a simpler alternative interface that reads from a configuration file:

.. code-block:: python

   import json
   import argparse
   from Granny.Interfaces.UI.GrannyUI import GrannyUI
   from Granny.Analyses.Analysis import Analysis


   class GrannyConfig(GrannyUI):
       """
       Configuration file-based interface for Granny.

       Reads analysis parameters from a JSON configuration file
       and executes batch analyses.
       """

       def __init__(self, parser: argparse.ArgumentParser):
           super().__init__(parser)

       def addProgramArgs(self) -> None:
           """Add config-specific arguments."""
           iface_grp = self.parser.add_argument_group("Config interface args")

           iface_grp.add_argument(
               "--config",
               dest="config_file",
               type=str,
               required=True,
               help="Path to JSON configuration file",
           )

       def run(self):
           """Execute analyses based on config file."""
           args, _ = self.parser.parse_known_args()

           # Load configuration
           with open(args.config_file, 'r') as f:
               config = json.load(f)

           # Process each analysis in config
           for analysis_config in config.get('analyses', []):
               analysis_name = analysis_config.get('type')
               params = analysis_config.get('parameters', {})

               # Find and instantiate analysis
               analysis = self._get_analysis(analysis_name)
               if not analysis:
                   print(f"Error: Unknown analysis '{analysis_name}'")
                   continue

               # Set parameters
               self._set_params(analysis, params)

               # Run analysis
               print(f"Running {analysis_name}...")
               results = analysis.performAnalysis()
               print(f"Completed: {len(results)} images processed\n")

       def _get_analysis(self, name: str) -> Analysis:
           """Get analysis by name."""
           for cls in Analysis.__subclasses__():
               if cls.__analysis_name__ == name:
                   return cls()
           return None

       def _set_params(self, analysis: Analysis, params: dict):
           """Set analysis parameters from dict."""
           analysis_params = analysis.getInParams()
           analysis.resetInParams()

           for param in analysis_params.values():
               label = param.getLabel()
               if label in params:
                   param.setValue(params[label])
               analysis.addInParam(param)

Example configuration file (``config.json``):

.. code-block:: json

   {
       "analyses": [
           {
               "type": "segmentation",
               "parameters": {
                   "input": "./images/raw",
                   "model": "pome_fruit-v1_0",
                   "confidence": 0.3
               }
           },
           {
               "type": "starch",
               "parameters": {
                   "input": "./results/segmentation",
                   "starch_threshold": 175,
                   "blur_kernel": 9
               }
           }
       ]
   }

Best Practices
--------------

Argument Parsing
~~~~~~~~~~~~~~~~

- Use ``add_argument_group()`` to organize interface-specific arguments
- Provide sensible defaults for optional arguments
- Include helpful descriptions in the ``help`` parameter
- Use appropriate types (int, str, float, bool) for type checking

Error Handling
~~~~~~~~~~~~~~

- Validate user input before passing to analyses
- Provide clear error messages for common mistakes
- Gracefully handle missing dependencies
- Log errors appropriately for the interface type

User Feedback
~~~~~~~~~~~~~

- Provide progress updates during long operations
- Display parameter values being used (especially defaults)
- Show clear success/failure messages
- For GUIs, use progress bars and status messages

Parameter Discovery
~~~~~~~~~~~~~~~~~~~

- Use ``Analysis.__subclasses__()`` to discover available analyses
- Use ``analysis.getInParams()`` to discover analysis parameters
- Dynamically generate UI elements based on parameter types
- Respect parameter constraints (min/max, required/optional)

Testing
-------

Create test file ``tests/test_Interfaces/test_GrannyWeb.py``:

.. code-block:: python

   from Granny.Interfaces.UI.GrannyWeb import GrannyWeb
   import argparse


   def test_init():
       """Test interface initialization."""
       parser = argparse.ArgumentParser()
       interface = GrannyWeb(parser)
       assert interface is not None


   def test_addProgramArgs():
       """Test argument addition."""
       parser = argparse.ArgumentParser()
       interface = GrannyWeb(parser)
       interface.addProgramArgs()

       # Verify arguments were added
       args = parser.parse_args(['--port', '8080'])
       assert args.port == 8080

Advanced Topics
---------------

Dynamic Parameter UI Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GUI or web interfaces, generate form fields based on parameter types:

.. code-block:: python

   def generate_parameter_fields(analysis: Analysis) -> list:
       """
       Generate UI field specifications for analysis parameters.

       Returns:
           List of field specifications for UI rendering
       """
       fields = []
       params = analysis.getInParams()

       for param in params.values():
           field = {
               "name": param.getLabel(),
               "label": param.getLabel().replace('_', ' ').title(),
               "type": param.getType().__name__,
               "default": param.getValue(),
               "required": param.getIsRequired(),
               "help": param.getHelp()
           }

           # Add constraints for numeric types
           if hasattr(param, 'getMin'):
               field["min"] = param.getMin()
               field["max"] = param.getMax()

           fields.append(field)

       return fields

Asynchronous Processing
~~~~~~~~~~~~~~~~~~~~~~~~

For web or GUI interfaces, run analyses asynchronously:

.. code-block:: python

   import threading
   from queue import Queue


   def run_analysis_async(analysis: Analysis, result_queue: Queue):
       """Run analysis in background thread."""
       try:
           results = analysis.performAnalysis()
           result_queue.put({"success": True, "results": results})
       except Exception as e:
           result_queue.put({"success": False, "error": str(e)})

Multi-Analysis Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced interfaces, implement workflow management:

.. code-block:: python

   from Granny.Interfaces.Scheduler.Scheduler import Scheduler


   def run_workflow(workflow: list) -> list:
       """
       Execute a multi-step analysis workflow.

       Args:
           workflow: List of (analysis, dependencies) tuples

       Returns:
           List of all results
       """
       scheduler = Scheduler()

       for analysis, deps in workflow:
           scheduler.add_analysis(analysis, deps)

       return scheduler.schedule()

Troubleshooting
---------------

**Interface not available in CLI:**

- Verify you added your interface to the choices in ``GrannyBase.py``
- Check the import statement is correct
- Ensure the ``elif`` branch for your interface exists

**Parameters not being set:**

- Verify you're calling ``resetInParams()`` before setting parameters
- Check that parameter names match between UI and analysis
- Ensure type conversion is correct (str → int/float where needed)
- Verify you're calling ``addInParam()`` after setting values

**Analysis not found:**

- Check that you're importing all analysis classes
- Verify ``Analysis.__subclasses__()`` returns expected classes
- Ensure analysis ``__analysis_name__`` matches what user provides

Next Steps
----------

- Study ``Granny/Interfaces/UI/GrannyCLI.py`` as a reference implementation
- Review ``Granny/Interfaces/UI/GrannyPyQt.py`` for GUI patterns
- Refer to :doc:`api_reference` for detailed API documentation
- Test your interface with all available analyses
