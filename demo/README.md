## Quick start 
Don't want to install anything on your computer, we have set up a [Google Colab](soon). 

## Intallation and Usage
Want to try on your dataset? You can install our model to get started. First, clone to your directory: 

```bash 
git clone https://github.com/SystemsGenetics/granny.git && cd granny
```

It is recommended to use a package manager such as [conda](https://www.anaconda.com/) or [virtualenv](https://pypi.org/project/virtualenv/) to create a seperate, independent environment for **GRANNY**. An description of the package installation using conda is provided below. 

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
sh setup.sh
```

### Command Line Interface
To perform extraction and/or rating on "Granny Smith" images, run **GRANNY** in the following way in the command line: 

```bash
granny  [-a,--action] <ACTION>  [-p,--image_path] <PATH>  [-m,--mode] <MODE>  [-v,--verbose] <VERBOSE>
```

where: 
| Arguments  | Type | Values |
| ---------  | ---- | ------ |
| `<ACTION>` | Required | Either **"extract"** or **"rate"**: <br />"extract" - perform individual apples extraction from apple trays; <br /> "rate" - perform disorder rating on single apple images.|
| `<PATH>`   | Required | Specify either an image directory containing multiple images or a single image file. |
| `<MODE>`   | Optional | Default to 1. Specify 2 for multiple images processing in `--action rate`. |
| `<VERBOSE>` | Optional | Default to 1. Specify 0 to turn off model display. |

### Example 
This is an apple tray, consisting of 18 apples: 

<div align="center">
  <img src="images/apple_tray/6moPos_TC-1(3)-2-A.JPG" width="500px" />
  <p>Example of an apple tray.</p>
</div>


In the command line, run GRANNY 
```bash 
granny --action extract --path images/apple_tray/6moPos_TC-1(3)-2-A.JPG
```
to get a full-tray mask: 

<div align="center">
  <img src="images/full_masked_images/6moPos_TC-1(3)-2-A.png" width="500px" />
  <p> </p>
</div>


... and individual apples:

1st row: 
<p float="left">
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_4.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_3.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_2.png" width="100" /> 
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_1.png" width="100" />
</p>

2nd row: 
<p float="left">
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_9.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_8.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_7.png" width="100" /> 
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_6.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_5.png" width="100" />
</p>

3rd row:
<p float="left">
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_13.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_12.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_11.png" width="100" /> 
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_10.png" width="100" />
</p>


4th row: 
<p float="left">
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_18.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_17.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_16.png" width="100" /> 
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_15.png" width="100" />
    <img src="images/segmented_images/6moPos_TC-1(3)-2-A_14.png" width="100" />
</p>

### Output and Results
Upon completion, **GRANNY** will have created a directory named `results` by default. Depending on the `<ACTION>`, output images will be stored in the corresponding sub-directories:


- `full_masked_images` - stores masked full-tray images 

- `segmented_images` - stores individual apple images

- `binarized_images` - 

| `<ACTION>` | Location | 
| ---------- | -------- | 
| "extract"  |  `full_masked_images` - stores original images with detected objects. <br /> `segmented_images` - stores each object instance as an individual image. | 
| "rate"     |  `binarized_images` - stores processed images with the scald being removed.|  


### **GRANNY** Directory
Please refer to [**GRANNY** directory description](https://github.com/SystemsGenetics/granny/blob/master/GRANNY/README.md) for a detailed listing of the **GRANNY**'s components.  