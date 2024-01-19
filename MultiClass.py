import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, ViTModel, AutoImageProcessor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoProcessor, CLIPVisionModel
import os
from PIL import Image
from transformers import AutoTokenizer, GPT2LMHeadModel

"""
Model class for MultiGPT

Limited Args as of now, TODO add greater functionality 

"""

class MultimodalGPTViT(nn.Module):
    def __init__(self, text_model_name, vision_model_name, vision_embedding_size=768):
        super(MultimodalGPTViT, self).__init__()
        #Intitalize both text and image models
        self.text_model = GPT2LMHeadModel.from_pretrained(text_model_name,torch_dtype=torch.float32)
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name,torch_dtype=torch.float32)#ViTModel.from_pretrained(vision_model_name,torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.image_preprocess = AutoProcessor.from_pretrained(vision_model_name)
        #Intialize project layer, norm, and img transformer
        text_embedding_size = self.text_model.config.hidden_size
        self.vision_projection = nn.Linear(vision_embedding_size, text_embedding_size)
        self.norm_layer = nn.LayerNorm(text_embedding_size)
        self.image_trans = nn.TransformerEncoderLayer(d_model=text_embedding_size, nhead=8)

        #Freeze image model
        for param in self.vision_model.parameters(): param.requires_grad = False



    def forward(self, input_ids, attention_mask, processed_image = None):
        if processed_image != None:
            pixel_values = processed_image['pixel_values'].squeeze(1)
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            #Get inital token embeddings
            token_embeddings = self.text_model.transformer.wte.weight[input_ids,:]

            vision_outputs = self.vision_model(pixel_values)
            vision_embedding = vision_outputs.last_hidden_state # Get the last layer

            # Project vision embedding to match text embedding size, normalize, and pass through transformer
            projected_vision_embedding = self.vision_projection(vision_embedding)
            projected_vision_embedding = self.norm_layer(projected_vision_embedding)
            projected_vision_embedding = self.image_trans(projected_vision_embedding)
            # Add the 1-51 token's embedding with the projected vision embedding + image token TODO This is hard coded for prof of concep
            token_embeddings[:, 1:51, :] = token_embeddings[:, 1:51, :] + projected_vision_embedding

            # Forward pass through the language model
            encoder_outputs = self.text_model(inputs_embeds=token_embeddings, attention_mask=attention_mask)
        else:
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            encoder_outputs = self.text_model(input_ids = input_ids, attention_mask=attention_mask)
        return encoder_outputs
    #Basic text generation function
    def generate_text(self, input_text, input_images=None, max_length=50, **generation_kwargs):
        # Tokenize text input
        input_encodings = self.tokenizer(input_text, return_tensors='pt',add_special_tokens = True)
        if input_images is not None:
            # Process and integrate image input
            processed_image = self.image_preprocess(images = input_images, return_tensors = 'pt')
            vision_outputs = self.vision_model(processed_image['pixel_values'])
            vision_embedding = vision_outputs.last_hidden_state
            projected_vision_embedding = self.vision_projection(vision_embedding)
            projected_vision_embedding = self.norm_layer(projected_vision_embedding)
            projected_vision_embedding = self.image_trans(projected_vision_embedding)
            # Replace the first token's embedding with the projected vision embedding
            token_embeddings = self.text_model.transformer.wte.weight[input_encodings['input_ids'],:]


            token_embeddings[:, 1:51, :] = token_embeddings[:, 1:51, :] + projected_vision_embedding
            # Generate text with the modified embeddings
            generated_ids = self.text_model.generate(inputs_embeds=token_embeddings, max_length=max_length, attention_mask = input_encodings['attention_mask'], **generation_kwargs)
        else:
            # Generate text without image input
            generated_ids = self.text_model.generate(input_ids=input_encodings['input_ids'], max_length=max_length, **generation_kwargs)

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    #Functions for tensor operations during training
    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    def get_text_model_vocab_size(self):
        return self.text_model.config.vocab_size
    
    def get_tokenizer(self):
        return self.tokenizer











"""Dataset Class For MultiGPT
Preprocessing is hardcoded in #TODO maybe change
"""


class TextImageDataset_GPT(Dataset):
    def __init__(self, csv_file, root_dir,low,high):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            high (int): Highest index for dataset
            low  (int): Lowest index for dataset
        """
        self.root_dir = root_dir
        self.annotations = pd.read_csv(os.path.join(self.root_dir,csv_file)).iloc[low:high]
        self.image_preprocess = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")#AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.max_length = 512

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
            if isinstance(self.annotations.iloc[idx, 0],str):
                img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
                image = Image.open(img_name)
                if image.mode != 'RGB':
                    image = image.convert("RGB")
                processed_image = self.image_preprocess(images = image, return_tensors="pt")
            else:
                processed_image = []

            text = self.annotations.iloc[idx, 1] + '<|endoftext|>'
            input_encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt',add_special_tokens = True)
            sample = {'image': processed_image, 'input_ids': input_encodings['input_ids'], 'attention_mask': input_encodings['attention_mask']}
            return sample
