### Intallation and Usage
To get started, clone the model to your directory: 

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

### Command Line Arguments
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

### Output and Results
Upon completion, **GRANNY** will have created a directory named `results` by default. Depending on the `<ACTION>`, output images will be stored in the corresponding sub-directories:
 
- `binarized_images` - stores 

| `<ACTION>` | Location | 
| ---------- | -------- | 
| "extract"  |  `full_masked_images` - stores original images with detected objects. <br /> `segmented_images` - stores each object instance as an individual image. | 
| "rate"     |  `binarized_images` - stores processed images with the scald being removed.|  


### **GRANNY** Directory
Please refer to [**GRANNY** directory description](https://github.com/SystemsGenetics/granny/blob/master/GRANNY/README.md) for a detailed listing of the **GRANNY**'s components.  
