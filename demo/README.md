## Quick start
Don't want to install anything on your computer, we have set up a quick demo on [Google Colab](https://colab.research.google.com/drive/10aJ_kQMXNRb9srB-YD0OPJpE8GOvgQlU?usp=share_link).

--- 

## Example

This is an apple tray, consisting of 18 apples:

<div align="center">
  <img src="images/apple_tray/demo_image.JPG" width="500px" />
  <p>Example of an apple tray.</p>
</div>

1. In the command line, run Granny

```bash
granny --action extract --path images/apple_tray/demo_image.JPG
```

to get a full-tray mask:

<div align="center">
  <img src="images/full_masked_images/demo_image.png" width="500px" />
  <p> </p>
</div>

... and individual apples:

1st row:

<p float="left">
    <img src="images/segmented_images/demo_image_4.png" width="100" />
    <img src="images/segmented_images/demo_image_3.png" width="100" />
    <img src="images/segmented_images/demo_image_2.png" width="100" /> 
    <img src="images/segmented_images/demo_image_1.png" width="100" />
</p>

2nd row:

<p float="left">
    <img src="images/segmented_images/demo_image_9.png" width="100" />
    <img src="images/segmented_images/demo_image_8.png" width="100" />
    <img src="images/segmented_images/demo_image_7.png" width="100" /> 
    <img src="images/segmented_images/demo_image_6.png" width="100" />
    <img src="images/segmented_images/demo_image_5.png" width="100" />
</p>

3rd row:

<p float="left">
    <img src="images/segmented_images/demo_image_13.png" width="100" />
    <img src="images/segmented_images/demo_image_12.png" width="100" />
    <img src="images/segmented_images/demo_image_11.png" width="100" /> 
    <img src="images/segmented_images/demo_image_10.png" width="100" />
</p>

4th row:

<p float="left">
    <img src="images/segmented_images/demo_image_18.png" width="100" />
    <img src="images/segmented_images/demo_image_17.png" width="100" />
    <img src="images/segmented_images/demo_image_16.png" width="100" /> 
    <img src="images/segmented_images/demo_image_15.png" width="100" />
    <img src="images/segmented_images/demo_image_14.png" width="100" />
</p>

2. With individual apples extracted to your `"results"`, run Granny with a `"scald"` action

```bash
granny --action scald --image_dir ./results/segmented_images/ --num_instances 2
```

to get

1st row:

<p float="left">
    <img src="images/binarized_images/demo_image_4.png" width="100" />
    <img src="images/binarized_images/demo_image_3.png" width="100" />
    <img src="images/binarized_images/demo_image_2.png" width="100" /> 
    <img src="images/binarized_images/demo_image_1.png" width="100" />
</p>

2nd row:

<p float="left">
    <img src="images/binarized_images/demo_image_9.png" width="100" />
    <img src="images/binarized_images/demo_image_8.png" width="100" />
    <img src="images/binarized_images/demo_image_7.png" width="100" /> 
    <img src="images/binarized_images/demo_image_6.png" width="100" />
    <img src="images/binarized_images/demo_image_5.png" width="100" />
</p>

3rd row:

<p float="left">
    <img src="images/binarized_images/demo_image_13.png" width="100" />
    <img src="images/binarized_images/demo_image_12.png" width="100" />
    <img src="images/binarized_images/demo_image_11.png" width="100" /> 
    <img src="images/binarized_images/demo_image_10.png" width="100" />
</p>

4th row:

<p float="left">
    <img src="images/binarized_images/demo_image_18.png" width="100" />
    <img src="images/binarized_images/demo_image_17.png" width="100" />
    <img src="images/binarized_images/demo_image_16.png" width="100" /> 
    <img src="images/binarized_images/demo_image_15.png" width="100" />
    <img src="images/binarized_images/demo_image_14.png" width="100" />
</p>
