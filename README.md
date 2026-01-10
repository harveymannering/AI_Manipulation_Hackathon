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

* The [scores.csv](https://github.com/harveymannering/AI_Manipulation_Hackathon/blob/main/scores.csv) contains results from the following experiment: CelebA images are evaluated by three human preference models—ImageReward, PickScore, and HPS. For each image, the CSV includes the model scores and binary facial attributes. This allows us to analyze whether specific attributes correlate with higher or lower preference scores. For example, for the attractive attribute, the correlation coefficients are 0.039 (ImageReward), 0.466 (PickScore), and 0.401 (HPS). However, can these correlations be explained by other attributes? For instance, if we control for the smiling attribute, does the preference score still correlate with attractive?
* Finetune a diffusion model using a human preference model. Does this increase facial attractiveness?
    * We can use algorithms like ReFL or DRaFT to finetune the diffusion model found in the [Diffusion101 notebook](https://github.com/harveymannering/AI_Manipulation_Hackathon/blob/main/Diffusion101.ipynb).
    * Train a [ResNet](https://github.com/harveymannering/AI_Manipulation_Hackathon/blob/main/resnet.ipynb) on the [CelebA-HQ dataset](https://www.kaggle.com/datasets/ipythonx/celebamaskhq) resized to $256\times256$ to predict attrivativenss. Each image has a binary attractive label.
    * Generate 1,000 face images before and after finetuning. Use the classifier to count the number of attractive faces. Does finetuning increase or decrease this number?
* Do Text-to-Image models (e.g. Nano Banana, Stable Diffusion) already contain beauty bias? How would we evaluate this?
* Do vision–language models (VLMs) contain beauty bias? If a model such as QwenVL is asked to identify which images in the [CelebA-HQ dataset](https://www.kaggle.com/datasets/ipythonx/celebamaskhq) or the [Facial Beauty Rating dataset](https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores) are AI-generated, is it more likely to classify more attractive faces as real?
* **Literature Review**
    * Are there safety concerns related to beauty bias in other technologies?
    * Are there studies on how beauty standards shaped by Photoshop or social media affect people?
    * How do media-driven beauty standards impact self-image?
    * Are people more likely to trust attractive faces?
    * How do human preference models [ImageReward](https://arxiv.org/abs/2304.05977), [HPS](https://arxiv.org/abs/2306.09341), and [PickScore](https://arxiv.org/abs/2305.01569) gather their datasets?
