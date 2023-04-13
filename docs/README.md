## Quick start

This is just a boring documentation page explaining and providing instructions to the user, you can check our [demo](https://github.com/SystemsGenetics/granny/tree/master/demo) page for specific usage of each command.

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
| `<ACTION>` | Required | **"extract"**: do [Extraction of Fruits](https://github.com/SystemsGenetics/granny/tree/master/demo#extract) <br /> **"scald"**: do [Superficial Scald](https://github.com/SystemsGenetics/granny/tree/master/demo#scald) <br /> **"starch"**: do [Cross-section Starch](https://github.com/SystemsGenetics/granny/tree/master/demo#starch) <br /> **"pear"**: do [Pear Color Sorting](https://github.com/SystemsGenetics/granny/tree/master/demo#pear)|
| `<PATH>` | Required | Specify either an image directory containing multiple images or a single image file. |
| `<NUM_INSTANCES>` | Optional | Default to 18. Specify 2 for multiple images processing in `--action scald`. |
| `<VERBOSE>` | Optional | Default to 1. Specify 0 to turn off model display. |

---

## <a name="output"></a> Output and Results

Upon completion, **Granny** will create a directory named `results` by default. Depending on the `<ACTION>`, output images will be stored in the corresponding sub-directories:

- `full_masked_images` - stores masked full-tray images

- `segmented_images` - stores individual apple images

- `binarized_images` - stores non-scald individual apple images

| `<ACTION>` | Directory                                                                                                                                            |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| "extract"  | `full_masked_images` - stores original images with detected objects. <br /> `segmented_images` - stores each object instance as an individual image. |
| "scald"    | `binarized_images` - stores processed images with the scald being removed.                                                                           |

---

## <a name="granny-dir"></a> **Granny** Code Directory

Please refer to [**Granny** directory description](https://github.com/SystemsGenetics/granny/blob/master/GRANNY/README.md) for a detailed listing of the **Granny**'s components.
