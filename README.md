# Before you can use the renderer in pyrender/

# Use colmap to extract points from a sequence of images.

## See how to do so [here](https://colmap.github.io/), and save the sparse and dense reconstructions as text.

# Generate a scene from a COLMAP
After converting something to a scene using COLMAP, saving both the sparse and dense point clouds as text, run convert_colmap.py from .//utils with the following arguments:

1 -> image folder
2 -> dense folder
3 -> sparse folder
4 -> 0/1/2 (nothing/environment+points/just environment (just use 1 initially))
4 -> 1/0   (images (just use 1 initially))
5 -> 1/0   (whole/patch (just use 1 initially))
6 -> output folder
7 (optional) -> environment type
8 (optional) -> mask folder
your command should look something like

```python utils/convert_colmap.py dataset_kitty_03/images dataset_kitty_03/sparse/txt dataset_kitty_03/dense/txt 1 1 1 /scene_kitti_03 1 dataset_kitty_03/masks```

# Generate a workspace with that scene.

Described in [pyrender/readme.md](/pyrender/readme.md).