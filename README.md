## COLA Data

### Multi-objects setting
The `data/COLA_multiobjects_matching_benchmark.json` file conatins the multi-objects setting validation. 

The json file contains data like:
`[link to image 1, caption 1, link to image 2, caption 2]`

The image links are in the format: https://cs-people.bu.edu/array/data/vg_gqa_images/2414605.jpg. `2414605` is the visual genome id.

The image 1 and image 2 have attributes and objects that are swapped in the captions.

Caption 1 applies to image 1 and not to image 2, and vice versa. 


### Single Objects setting

- `data/COLA_singleobjects_benchmark_GQA.json` contains the single-objects setting validation data for GQA.
- `data/COLA_singleobjects_benchmark_CLEVR.json` contains the single-objects setting validation data for CLEVR.
- `data/COLA_singleobjects_benchmark_PACO.json` contains the single-objects setting validation data for PACO.

The JSON contains two keys: `labels` and `data`. 

The `labels` contains the names of the 320 multi-attribute object classes for the benchmark.

The `data` is a list with each entry in the format:

`[image_file, objects_attributes_annotation, label, hard_list]`

- `image_file`: Path to Visual Genome Image
- `objects_attributes_annotation`: a dictionary of the objects in the image along with the attributes. This is absent for the PACO verson.
- `label`: `0` or `1` label of whether each of the 320 classes in the `labels` is present in the image or not. 
- `hard_list`: `0` or `1` label of whether the image is counted within a difficult set for each of the 320 class labels. 
For instance, if a class label is "square white plate", a hard image could be a images with something white or square or plate. Rsults in the main COLA paper are on this hard set of images. 
Measure MAP on the set of images with this label as 1 for each class for measuring the COLA paper MAPs.  


