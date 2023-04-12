## Quick start
This is just a boring documentation page explaining and providing instructions to the user, we suggest you check our [demo](https://github.com/SystemsGenetics/granny/tree/master/demo) page before going over this page.

---

## Contents

1. [Installation](#installation)
2. [Command Line Interface](#cli)
3. [Output Directory](#output)
4. [Granny Code Directory](#granny-dir)

--- 

## <a name="installation"></a> Installation

Want to try on your dataset? You can install our model to get started. First, it is recommended to use a package manager such as [conda](https://www.anaconda.com/) or [virtualenv](https://pypi.org/project/virtualenv/) to create a seperate, independent environment for **Granny**. An description of the package installation using conda is provided below.

Due to the limitation of TensorFlow 1.15, it is required to have Python version be **less than or equal** to 3.7

```bash
conda create -n <venv> python==3.7 -y
```

where `<venv>` is the name of the virtual environment

To activate the environment:

```bash
conda activate <venv>
```

Inside the environment, run the following to set up command line interfaces:

```bash
pip install --upgrade granny
```

---

## <a name="cli"></a> Command Line Interface

To perform extraction and/or rating on "Granny Smith" images, run **Granny** in the following way in the command line:

```bash
granny  [-a,--action] <ACTION>
        [-d,--image_dir] <PATH>  
        [-n,--num_instances] <NUM_INSTANCES>  
        [-v,--verbose] <VERBOSE>
```

where:
| Arguments | Type | Values |
| --------- | ---- | ------ |
| `<ACTION>` | Required | Either **"extract"** or **"rate"**: <br />"extract" - perform individual apples extraction from apple trays; <br /> "rate" - perform disorder rating on single apple images.|
| `<PATH>` | Required | Specify either an image directory containing multiple images or a single image file. |
| `<NUM_INSTANCES>` | Optional | Default to 18. Specify 2 for multiple images processing in `--action rate`. |
| `<VERBOSE>` | Optional | Default to 1. Specify 0 to turn off model display. |

---

## <a name="output"></a> Output and Results

Upon completion, **Granny** will have created a directory named `results` by default. Depending on the `<ACTION>`, output images will be stored in the corresponding sub-directories:

- `full_masked_images` - stores masked full-tray images

- `segmented_images` - stores individual apple images

- `binarized_images` - stores non-scald individual apple images


| `<ACTION>` | Location                                                                                                                                             |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| "extract"  | `full_masked_images` - stores original images with detected objects. <br /> `segmented_images` - stores each object instance as an individual image. |
| "rate"     | `binarized_images` - stores processed images with the scald being removed.                                                                           |

---

## <a name="granny-dir"></a> **Granny** Code Directory

Please refer to [**Granny** directory description](https://github.com/SystemsGenetics/granny/blob/master/GRANNY/README.md) for a detailed listing of the **Granny**'s components.
