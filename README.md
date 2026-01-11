# Biased Attractiveness Bench

The repo contains code for Biased Attractiveness Bench (BAB), a dataset of 300 images with varying levels of attrativeness and corresponding label. Images are generated using FLUX.1 Kontext and labelled using SwinFace. Images can be found in the [generated_images_flux](https://github.com/harveymannering/AI_Manipulation_Hackathon/tree/main/generated_images_flux) directory. To run our generation code run the following command:

``` 
python generate_images.py --output_dir ./generated_images/ --cache_dir ./cache_dir/
```

To measure the attrativeness baises for the models we consider run the following

```
python measure_bias.py --image_path ./generated_images_flux --dataset synthetic --caption_strategy const --batch_size 1 --const_caption "a portrait photo of a face"
```
