Developer's Guide
=================

Welcome to the Granny Developer's Guide! This guide is designed for Python developers who want to extend Granny by creating new analysis modules or user interfaces.

Overview
--------

Granny follows a Model-View-Controller (MVC) architectural pattern, making it highly extensible through a pluggable module system. The framework provides:

- **Analysis modules** - Pluggable rating algorithms for different fruit quality assessments
- **Interface modules** - Multiple ways for users to interact with Granny (CLI, GUI, mobile)
- **Value system** - Type-safe parameter handling with automatic CLI argument generation
- **Scheduler** - Dependency management for chaining analyses together
- **Image I/O** - Consistent handling of images and metadata

Key Extension Points
--------------------

There are two primary ways to extend Granny:

1. **Adding New Analysis Modules**

   Create new fruit quality assessment algorithms by inheriting from the ``Analysis`` base class. Your analysis will automatically integrate with all Granny interfaces and benefit from the parameter system, metadata tracking, and result storage.

   Examples: StarchArea, BlushColor, PeelColor, SuperficialScald, Segmentation

2. **Adding New Interfaces**

   Create new ways for users to interact with Granny by inheriting from the ``GrannyUI`` base class. Your interface can leverage any existing analysis module and will automatically receive their parameters.

   Examples: GrannyCLI (command-line), GrannyPyQt (desktop GUI), Mobile interface (future)

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2

   adding_analysis
   adding_interface
   api_reference
   example_analysis
   bugfixes

Getting Started
---------------

**For adding a new analysis module:**

1. Read :doc:`adding_analysis` for step-by-step instructions
2. Review :doc:`example_analysis` for a complete working example
3. Refer to :doc:`api_reference` for detailed API documentation

**For adding a new interface:**

1. Read :doc:`adding_interface` for step-by-step instructions
2. Study ``Granny/Interfaces/UI/GrannyCLI.py`` as a reference implementation
3. Refer to :doc:`api_reference` for detailed API documentation

Development Prerequisites
--------------------------

- Python >= 3.9
- Familiarity with object-oriented programming
- Understanding of abstract base classes
- Experience with NumPy and OpenCV (for analysis modules)
- Knowledge of argparse or PyQt (for interface modules)

Installation for Development
-----------------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/SystemsGenetics/granny.git
   cd granny
   pip install -e .

This allows you to modify the code and see changes immediately without reinstalling.

Testing Your Extensions
-----------------------

Granny uses pytest for testing. After adding your module:

1. Create test files in ``tests/`` matching your module structure
2. Run tests: ``python -m pytest``
3. Test specific modules: ``python -m pytest tests/test_Analyses/test_YourAnalysis.py``

Contributing
------------

If you create a useful analysis module or interface, consider contributing it back to the project:

1. Ensure your code follows the existing patterns
2. Add comprehensive docstrings
3. Include tests for your module
4. Update documentation as needed
5. Submit a pull request on GitHub

Support
-------

For questions about extending Granny:

- Review the existing analysis modules in ``Granny/Analyses/``
- Check the API reference documentation
- Open an issue on GitHub: https://github.com/SystemsGenetics/granny/issues
