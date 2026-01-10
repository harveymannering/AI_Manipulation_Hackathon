# AI_Manipulation_Hackathon

## Intro

An exciting avenue currently being explored by diffusion model researchers is the use of human preference models to finetune diffusion models. Popular examples include [ImageReward](https://arxiv.org/abs/2304.05977), [HPS](https://arxiv.org/abs/2306.09341), and [PickScore](https://arxiv.org/abs/2305.01569). These models learn to approximate how humans judge the quality of AI-generated images. As with language models, they can be used for finetuning, via RLHF or supervised learning, to improve image generators. 

However, human preference models may encode biases. If so, they can enable reward hacking during fine-tuning: instead of learning to generate better images, diffusion models may learn to exploit weaknesses in the preference model to achieve higher scores (similar to sycophancy bias in LLMs).

Because these models are trained on human annotations, they may inherit unconscious human biases. For example [recent work](https://www.sciencedirect.com/science/article/pii/S0001691824005481) suggests that people are less likely to believe a face image is AI-generated if they perceive it as attractive. This raises the question: do human preference models learn a similar bias?

To investigate, we analyze the [Facial Beauty Rating dataset](https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores) with examples shown below:

<img width="794" height="400" alt="image" src="https://github.com/user-attachments/assets/6d2d058a-d269-4c26-8979-861fa4c6384c" />

We find that facial attractiveness correlates with the scores assigned by human preference models. This effect is particularly pronounced for the HPS model, as shown in the figure below.

<img width="2546" height="581" alt="image" src="https://github.com/user-attachments/assets/9df8bce6-01d0-4c82-a297-62022979cc88" />

How big of a problem is beauty bias? How widespread is the phenomenon? And can it lead to reward hacking?

## Tasks

* The [scores.csv](https://github.com/harveymannering/AI_Manipulation_Hackathon/blob/main/scores.csv) contains results for the following experiment. Input CelebA images in three huma preference models ImageReward, PickScore, and HPS. The score that these models give each image is provided in the csv along with binary attributes of each image. From this we can see if any attributes correlate with low or high human preference scores. For example, for the attrative attribute we see correlation coefficients of 0.039116, 0.466292, and 0.401465 for the ImageReward, PickScore and HPS, respectively. But can this correlation be explained using other attributes? For example, if we control for the smiling attribute, does the human preference score still correlate with the attrative attribute?
* Finetune a diffusion model using a human preference model. Does this increase the attractivenss of faces?
    * We can use algorithms like ReFL or DRaFT to finetune the diffusion model found in the [Diffusion101 notebook](https://github.com/harveymannering/AI_Manipulation_Hackathon/blob/main/Diffusion101.ipynb).
    * Train a [ResNet](https://github.com/harveymannering/AI_Manipulation_Hackathon/blob/main/resnet.ipynb) on the [CelebA-HQ dataset](https://www.kaggle.com/datasets/ipythonx/celebamaskhq) resized to $256\times256$ to predict attrivativenss. Each images in this dataset has a binary label called attractive.
    * Generate 1,000 face images with the diffusion model before and after finetuning. Use the classifier to count the number of attractive faces. Does finetuning increase or decrease the number of attractive faces?
* Do Text-to-Image models (e.g. Nano Banana, Stable Diffusion) already contain beauty bias? How would we evaluate this?
* Do VLMs contain beauty bais? If we ask a model like QwenVL to choose which images in [CelebA-HQ dataset](https://www.kaggle.com/datasets/ipythonx/celebamaskhq) or the [Facial Beauty Rating dataset](https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores) are AI generated, is it more likely to pick the more attractive faces as being real?
* **Literature Review** - are there safety concerns around beauty bias for other technologies? Are there paper around how beauty standard created by Photoshop or social media have effected people? How do beauty standard in media effect peoples self image? Are people more trusting off attractive faces? 
