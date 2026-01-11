# Biased Attractiveness Bench

The repo contains code for Biased Attractiveness Bench (BAB), a dataset of 300 images with varying levels of attrativeness and corresponding label. Images are generated using FLUX.1 Kontext and labelled using SwinFace. Images can be found in the [generated_images_flux](https://github.com/harveymannering/AI_Manipulation_Hackathon/tree/main/generated_images_flux) directory and labels can be found in the [result_flux.json](https://github.com/harveymannering/AI_Manipulation_Hackathon/blob/main/result_flux.json) file. To run our generation code run the following command:

``` 
python generate_images.py --output_dir ./generated_images/ --cache_dir ./cache_dir/
```

To measure the attrativeness baises for the models we consider run the following

```
python measure_bias.py --image_path ./generated_images_flux --dataset synthetic --caption_strategy const --batch_size 1 --const_caption "a portrait photo of a face"
```

To run VLM evaluation (current default is Qwen3-VL-8B), please run the following inside vlms directory
```
python vlm_eval.py
```
OR (for customised run)
```
python vlm_eval.py --seed 42 --modelname "Qwen/Qwen3-VL-8B-Instruct" --datasetname "flux"
```

This work was done as a part of the AI Manipulation Hackathon at Apart Research from Jan 9th to Jan 11th 2026.
