The Command-Line Interface
==========================

The command-line is a fully text-based interface and can be used to execute all of analytical tools of Granny.  While the command-line is not the most intuitive interface for those of us used to the graphical desktop of a Windows or Mac computer, the command-line provides some important benefits:

- It is intuitive for research scientists used to the Linux command-line
- Supports use on modern high-performance computing resources
- Supports the ability to organize steps into a workflow.

The following describes how to use Granny on the command-line.  Regardless if you are on a Linux, Mac OS X, or Windows terminal, the commands for Granny will be the same. The Granny command-line is not available on the mobile version of the software.

Execute Granny
--------------
Once Granny is installed, you can simply execute it by calling the program and providing arguments. A simple command to run Granny is to ask it for its version. To see the version of Granny that you have installed type the following into the terminal window and press enter:

.. code:: bash

    granny -v

You should see output similar to the following:

::

    Granny 1.0a0

Program Arguments
-----------------

The `-v` value that was used in the previous step when running the `granny` program is an **argument**. Arguments tell the Granny program what to do. In the example above, the `-v` argument asks granny to simply print out the version number. 

Granny requires that you always tell it what interface you want to use. You need to provide the `-i` (short form) or `--interface` argument (long form) and indicate that you want to use the command-line interface by giving the value `cli`:

.. code:: bash

    granny -i cli

or 

.. code:: bash

    granny -interface cli

Alternatively, you can put values of arguments in quotes:

.. code:: bash

    granny -i "cli"

or 

.. code:: bash

    granny -interface "cli"

Using quotes around values is helpful if the values have spaces or other non alpha-numeric characters that you want included in the value.  The command-line will treat spaces as a separation between values (rather than a space) and other characters may have different meaning on the command-line.  Therefore, quotes may be needed.

Get Help
--------
Aside from this User's Guide, Granny does provide some assistance when running on the command-line to remind you of available arguments. If you run Granny without any arguments, or if you specify incorrect arguments, you will receive help text to remind you how to run Granny and what arguments are available. For example running Granny without arguments:

.. code:: bash

    granny

Should print similar output:

:: 

    error: the following arguments are required: -i/--interface
    usage: granny -i {cli,gui} [-h] [-v]


    options:
    -i {cli,gui}, --interface {cli,gui}
                            Indicates the user interface to use, either the command-line (cli) or the graphical interface (gui).
    -h, --help            show this help message and exit
    -v, --version         show program's version number and exit


Notice the message informs you of three things:

1. It tells you that you are missing the interface (`-i`) argument
2. It gives you instructions for how to run Granny
3. It gives valid options the arguments.

For the `usage:` section, the notation has the following meaning:

- `[]`: arguments surrounded by square brackets mean that the argument is not required but can be provided.
- `{}`: curly braces indicate which values for an argument are valid.


Finding Analysis Modules
------------------------

If you correctly specify the interface but provide no other arguments you will receive a message indicating that the `analysis` argument is required. For example:

.. code:: bash

    granny -i cli

You will see the following:

::

    error: the following arguments are required: --analysis
    usage: granny -i {cli,gui} [-v] --analysis {segmentation,blush,color,scald,starch}


    options:
    -i {cli,gui}          Indicates the user interface to use, either the command-line (cli) or the graphical interface (gui).
    -v                    show program's version number and exit

    CLI interface args:
    --analysis {segmentation,blush,color,scald,starch}
                            Indicates the analysis to run.


The message tells you which analysis modules are available to run. The analysis module names are listed in the curly braces (e.g.:  `{segmentation,blush,color,scald,starch}`).  You can run an analysis by providing its name after the `analysis`` argument.  For example:

.. code:: bash

  granny -i cli --analysis segmentation


Finding Analysis Arguments
--------------------------
If you know the analysis module you want to run, but you do not know what arguments you are allowed to provide, you can run granny and specify the analysis but provide no other arguments. For example, to get information about which which arguments are available for the `segmentation` analysis module run the following:

.. code:: bash

    granny -i cli --analysis segmentation

You should see the following:


::

    error: the following arguments are required: --input
    usage: granny -i {cli,gui} [-v] --analysis {segmentation,blush,color,scald,starch} [--model MODEL] --input INPUT


    options:
    -i {cli,gui}          Indicates the user interface to use, either the command-line (cli) or the graphical interface (gui).
    -v                    show program's version number and exit

    CLI interface args:
    --analysis {segmentation,blush,color,scald,starch}
                            Indicates the analysis to run.

    segmentation args:
    --model MODEL         Specifies the model that should be used for segmentation to identify fruit. The model can be specified using a known model name
                            (e.g. 'pome_fruit-v1_0'), and Granny will automatically retrieve the model from the online https://osf.io. Otherwise the value
                            must be a path to where the model is stored on the local file system. If no model is specified then the default model will be
                            used.
    --input INPUT         The directory where input images are located.


The output shows that you can specify two arguments for segmentation that include the `--model` and the `--input` arguments.