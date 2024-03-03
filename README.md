# GPT2MultiModal
Implementation of Multimodal GPT2

## Model Design ##

<img src="https://github.com/JoeyNiestroy/GPT2MultiModal/assets/106636917/5598c9cf-e749-43f3-a85b-50c81b184a72" width="800">

A pretrained CLIP-Vit-base model and GPT2-large model are the base vison and LM. Input images are proccessed first by the CLIP model then normalized and pass to a single transfomer layer before being added to the token embeddings. A special `<image>` token/embedding was added to the vocab and typcal inputs are formated as such " Q:  (50*`<image>`) How many people are in this image? A: 3 " The image token is repeated to match the shape of the image embeddings so that the two tensors can be added together. 

The total paramaters of the model are **<1 Billion**

##  Training Methods ## 

First part of training used base causal language modeling objective, to avoid issues seen in early training only the answer tokens were used to calculate loss. A subset of the 2017 VQA dataset was used.  ${\color{green}\text{Currently 100k/100k samples, First Epoch}}$

The second part of training will include RL with PPO with a basic reward model based off if the model correctly identified the answer. Sentence similarity failed for automated reward modeling for this task (Words Red and Black are high similar tokens, but for visual question answering one is correct one is not) , so currently tagging data myself, and planning to adapt model for automating this process.  ${\color{red} \text{Up-coming} }$

Finally the model will be trained off LLava Dataset for desciptive and longer form replies ${\color{red} \text{Up-coming} }$

## Example Model Outputs ## 

### All Examples outside of training data / Unseen by model ###

#### Input  Image ####
<img src="https://github.com/JoeyNiestroy/GPT2MultiModal/assets/106636917/095145ac-9be4-4cae-bc1f-8855ef8315ae" width="500">

#### Input ####
Q: What is happening in the image? 
#### Output ####
A: snowboarding on mountainside. no one skiing

#### Input  Image ####
<img src="https://github.com/JoeyNiestroy/GPT2MultiModal/assets/106636917/743e9b28-4188-4e7b-a567-cd8100b5efdb" width="500">

#### Input ####
Q: What color is the dog? 
#### Output ####
A: brown

