from MultiClass import MultimodalGPTViT, TextImageDataset_GPT
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import argparse

"""
Traning Loop script

Loads model and dataset and runs through pytorch training loop


"""


parser = argparse.ArgumentParser(description='Train a Multimodal LLaMaViT model.')

# Currently unused, will implement for reproducablilty 

# parser.add_argument('--csv_file', type=str, default='training_data_subset.csv', help='Path to the CSV file containing the training data.')
# parser.add_argument('--root_dir', type=str, default='/media/joey/Elements', help='Root directory of the training data.')
# parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
# parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training.')
# parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer.')
# parser.add_argument('--model_path', type=str, default='multimodal_temp.pth', help='Path to save the model.')


parser.add_argument('--data_subset_range_low', type=int, default=0)
parser.add_argument('--data_subset_range_high', type=int, default=5000)
parser.add_argument('--tag', type=int, default=0)
# parser.add_argument('--learning_rate', type=float, default=1e-4)

# Parse the arguments
args = parser.parse_args()
low = args.data_subset_range_low
high = args.data_subset_range_high
tag = args.tag


#Model Prep


model = MultimodalGPTViT('gpt2', "openai/clip-vit-base-patch32")
model.load_state_dict(torch.load('/media/joey/Elements/multimodal_temp_gpt.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.half()
model.train()



#Training Data setup
csv_file = 'training_data_subset_vqa.csv'
root_dir = '/media/joey/Elements'
text_image_dataset = TextImageDataset_GPT(csv_file=csv_file, root_dir=root_dir, low = low , high  =  high)

batch_size = 1
data_loader = DataLoader(text_image_dataset, batch_size=batch_size, shuffle=True)



#Training parameters
num_epochs = 1
lr = 1e-4
gradient_accumulation_steps = 128
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)

#Don't have memory for these optimizers :(
# optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


total_batches = len(text_image_dataset) // batch_size

# Initialize the Training Bar
progress_bar = tqdm(total=total_batches * num_epochs, desc="Training Progress")
count = 0


batch_counter = 0



#Set up weight dictionary
tokenizer = model.get_tokenizer()
weights_vocab = torch.ones(len(tokenizer)).to(device)

#image token
weights_vocab[50257] = 0

#EOS token
weights_vocab[50256] = 0

#Space token
weights_vocab[220] = 1

#Answer token
weights_vocab[50259] = 0
import time

print("Data Loaded, Main Loop beginning")

for epoch in range(num_epochs):
    
    epoch_loss = 0.0
    num_batches = 0
    
    batch_counter = 0
    
    for batch in data_loader:
        # Conditional if image is not given for sample
        if isinstance(batch['image'],list):
            pass
        else:
            image_data = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.cuda.amp.autocast():
            #Checks if image is given
            if isinstance(batch['image'],list):
                outputs = model(input_ids, attention_mask) 
            else:
                outputs = model(input_ids, attention_mask, image_data)

            
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            #Loss is only taken for tokens involved in answer
            index = (shift_labels == 50259).nonzero(as_tuple=True)[2][0]

            shift_labels =  shift_labels[:,:,index:]

            shift_logits = shift_logits[:,index:,:]

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index= model.get_pad_token_id(),weight= weights_vocab)

            current_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            
        # Handle nan losses
        if torch.isnan(current_loss):
            pass
        else:
            
            torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
            current_loss.backward()
            if (batch_counter + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            batch_counter += 1
        
        epoch_loss += current_loss
        num_batches += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"Epoch": epoch + 1, "Loss": current_loss})

        if batch_counter % 1000 == 0:
            time.sleep(60)
        if batch_counter % 6000 == 0:
            torch.save(model.state_dict(), 'multimodal_temp.pth')
            print(batch_counter)
    
    
    average_loss = epoch_loss / num_batches
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')
optimizer.step()

#Saving model states after loop
torch.save(model.state_dict(), '/media/joey/Elements/multimodal_temp_gpt.pth')
torch.save(model.state_dict(), '/media/joey/Elements/multimodal_temp_gpt_'+str(high)+'.pth')

