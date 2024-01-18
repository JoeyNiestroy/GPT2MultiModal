# GPT2MultiModal
Implementation of Multimodal GPT2

## Model Design ##
![image](https://github.com/JoeyNiestroy/GPT2MultiModal/assets/106636917/5598c9cf-e749-43f3-a85b-50c81b184a72)

A pretrained CLIP Vit model and GPT2 model are the base vison and LM. Input images are proccessed first by the CLIP model then normalized and pass to a single transfomer layer before being added to the token embeddings. A special `<image>` token/embedding was added to the vocab and typcal inputs are formated as such " Q: How many people are in this image (50*`<image>`) A: 3 " The image token is repeated to match the shape of the image embeddings so that the two tensors can be added together. 

##  Training Methods ## 

First part of training used base causal language modeling objective, to avoid issues seen in early training only the answer tokens were used to calculate loss. A subset of the 2017 VQA dataset was used.  

The second part of training includes RL with PPO through the RLT library with a basic reward model based off if the model correctly identified the correct answer Ex: -1 if wrong 1 if correct. 

Finally the model was trained off LLava Dataset for desciptive and longer form replies

