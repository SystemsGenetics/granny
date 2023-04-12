## Quick start
This is just a boring documentation page explaining and providing instructions to the user, we suggest you check our [demo](https://github.com/SystemsGenetics/granny/tree/master/demo) page before going over this page.

---

## Content

1. [Command Line Interface](#cli)
2. [Output Directory](#output)
3. [Granny Code Directory](#granny-dir)

---

## <a name="cli"></a> Command Line Interface

**Granny** can be run by the following way in the command line:

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
