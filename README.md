# AI_Manipulation_Hackathon

## Intro

An exciting avenue for diffusion model researchers are currently exploring are is to use human preference models to finetune diffusion models. This include popular examples like ImageReward, HPS, and PickScore. These model take learn to approximate how good an AI generated images looks. Like with language model, these human preference models can then be used to in finetuning (either RLHF or supervised finetuning) to improve image generator. However, what if human preference model contain biases? If they did they may lead reward hacking in the finetuning processes, where instead of diffusion models learning to generate better images, they learn how to get a better score from human preference models.

These model are trained on human preference data, meaning they may have picked up unconsious biases from human annotators. For example, [recent work](https://www.sciencedirect.com/science/article/pii/S0001691824005481) suggest that humans are less likely to beleive an image of a face is AI generated if they perceive it as attractive. Could human preference models have picked up this bias? To answer this question we look at the [Facial Beauty Rating dataset](https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores) with examples shown below:

<img width="794" height="400" alt="image" src="https://github.com/user-attachments/assets/6d2d058a-d269-4c26-8979-861fa4c6384c" />

We fine the facial attractiveness of the face correlates with the score given by human preference model. Below we can see this particularlly bad for the HPS model.

<img width="2546" height="581" alt="image" src="https://github.com/user-attachments/assets/9df8bce6-01d0-4c82-a297-62022979cc88" />

How big of a problem is beauty bias? How widespread is the phenomenon? And can it lead to reward hacking?

## Tasks

- 
