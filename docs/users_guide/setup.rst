Step 1: Project Setup
=====================

Before running Granny, it is recommended to create a new directory on your file system for each "project". A Granny project will consist of the set of fruit tray images that you want to analyze and the results that Granny will generate.   You will perform both segmentation and fruit rating in the same project folder.

.. note::

    It is best practice to create new project directory every time you run Granny on a new set of images.


.. note::

    It is best practice to not run different sets of images in the same project directory.
    

For this tutorial we will use demo images that are part of the Granny source code repository. The User's Guide will provide links to download these images. To follow along, please create a directory named `demo` anywhere on your computer file system that you want to practice. In practice be sure to follow the best practices notifications above.  

You do not need to copy the image files you want to analyze into the project directory. Those images can be stored anywhere on your computer. The project directory will be used for housing all of Granny's results.

Command-line Example
--------------------
If you feel most comfortable using the file browser on your computer, you can use that to create the project directory. However, if you would like to use the command-line directly, use the `mkdir` command to make a new directory. You can open the command-line terminal and type the following to create the `demo` directory in your home directory:

.. code:: bash

    mkdir demo

Then, navigate to that directory using the 'change directory', `cd` command:

.. code:: bash

    cd demo

.. note::

    If you create the project directory using the command-line you will be able to find it using your computers' file browser.  