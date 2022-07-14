#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url

import os
import random
import numpy as np
import math
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange, tqdm

from RES_VAE import VAE as VAE
from vgg19 import VGG19


from clearml import StorageManager, Task
from clearml import Dataset as cmlDataset
task = Task.init(project_name='bogdoll/anomaly_detection_simon', task_name='cnn-vae')

task.set_base_docker(
            "nvcr.io/nvidia/pytorch:21.10-py3",
            docker_setup_bash_script="apt-get update && apt-get install -y libfreetype6-dev && apt-cache search freetype | grep dev && apt-get install -y python3-opencv",
            docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all"
            )
task.execute_remotely('docker', clone=False, exit_process=True)

# In[2]:


batch_size = 32
image_size = 256
lr = 1e-4
nepoch = 100
start_epoch = 0
#dataset_root = "/disk/vanishing_data/mb274/data/cityscapes/test/"
dataset_root = cmlDataset.get(dataset_name= 'cityscapes_train', dataset_project= 'bogdoll/anomaly_detection_simon').get_local_copy()
save_dir = os.getcwd()
model_name = "test_train"
load_checkpoint  = False


# In[3]:


use_cuda = torch.cuda.is_available()
GPU_indx  = 0
device = torch.device(GPU_indx if use_cuda else "cpu")


# In[4]:


class MyDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.total_data_dir = os.listdir(data_dir)
        self.transform = transform
        self.label = 0
    def __len__(self):
        return len(self.total_data_dir)

    def __getitem__(self, idx):
        loc = os.path.join(self.data_dir, self.total_data_dir[idx])
        rgb_img = Image.open(loc).convert('RGB')
        #rgb_img = rgb_img.resize([256, 256], Image.ANTIALIAS)
        tensor_image = self.transform(rgb_img)

        return (tensor_image, self.label)

def get_data_STL10(transform, batch_size, download = True, root=dataset_root):
    print("Loading trainset...")
    trainset = MyDataset(root, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print("Loading testset...")
    testset = MyDataset(root, transform=transform)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("Done!")

    return trainloader, testloader


# In[5]:


#OLD way of getting features and calculating loss - Not used

#create an empty layer that will simply record the feature map passed to it.
class GetFeatures(nn.Module):
    def __init__(self):
        super(GetFeatures, self).__init__()
        self.features = None
    def forward(self, x):
        self.features = x
        return x

#download the pre-trained weights of the VGG-19 and append them to an array of layers .
#we insert a GetFeatures layer after a relu layer.
#layers_deep controls how deep we go into the network
def get_feature_extractor(layers_deep = 7):
    C_net = models.vgg19(pretrained=True).to(device)
    C_net = C_net.eval()
    
    layers = []
    for i in range(layers_deep):
        layers.append(C_net.features[i])
        if isinstance(C_net.features[i], nn.ReLU):
            layers.append(GetFeatures())
    return nn.Sequential(*layers)

#this function calculates the L2 loss (MSE) on the feature maps copied by the layers_deep
#between the reconstructed image and the origional
def feature_loss(img, recon_data, feature_extractor):
    img_cat = torch.cat((img, torch.sigmoid(recon_data)), 0)
    out = feature_extractor(img_cat)
    loss = 0
    for i in range(len(feature_extractor)):
        if isinstance(feature_extractor[i], GetFeatures):
            loss += (feature_extractor[i].features[:(img.shape[0])] - feature_extractor[i].features[(img.shape[0]):]).pow(2).mean()
    return loss/(i+1)


# In[6]:


#Linear scaling the learning rate down
def lr_Linear(epoch_max, epoch, lr):
    lr_adj = ((epoch_max-epoch)/epoch_max)*lr
    set_lr(lr = lr_adj)

def set_lr(lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy_with_logits(recon, x)
    KL_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    loss = recon_loss + 0.01 * KL_loss
    return loss


# In[7]:


transform = T.Compose([T.Resize(image_size), T.ToTensor()])

trainloader, testloader = get_data_STL10(transform, batch_size, download=False, root=dataset_root)


# In[8]:


#get a test image batch from the testloader to visualise the reconstruction quality
dataiter = iter(testloader)
test_images, _ = dataiter.next()
test_images.shape


# In[9]:


plt.figure(figsize = (60,30))
out = vutils.make_grid(test_images[0:2])
#plt.imshow(out.numpy().transpose((1, 2, 0)))


# In[10]:


#Create the feature loss module

# load the state dict for vgg19
state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
# manually create the feature extractor from vgg19
feature_extractor = VGG19(channel_in=3)

# loop through the loaded state dict and our vgg19 features net,
# loop will stop when net.parameters() runs out - so we never get to the "classifier" part of vgg
for ((name, source_param), target_param) in zip(state_dict.items(), feature_extractor.parameters()):
    target_param.data = source_param.data
    target_param.requires_grad = False
    
feature_extractor = feature_extractor.to(device)


# In[11]:


#Create VAE network
vae_net = VAE(channel_in=3, ch=256).to(device)
# setup optimizer
optimizer = optim.Adam(vae_net.parameters(), lr=lr, betas=(0.5, 0.999))
#Loss function
BCE_Loss = nn.BCEWithLogitsLoss()
loss_log = []


# In[12]:


#Create the save directory if it does note exist
if not os.path.isdir(save_dir + "/Models"):
    os.makedirs(save_dir + "/Models")
if not os.path.isdir(save_dir + "/Results"):
    os.makedirs(save_dir + "/Results")

if load_checkpoint:
    checkpoint = torch.load(save_dir + "/Models/" + model_name + "_" + str(image_size) + ".pt", map_location = "cpu")
    print("Checkpoint loaded")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    vae_net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint["epoch"]
    loss_log = checkpoint["loss_log"]
else:
    #If checkpoint does exist raise an error to prevent accidental overwriting
    if os.path.isfile(save_dir + "/Models/" + model_name + "_" + str(image_size) + ".pt"):
        raise ValueError("Warning Checkpoint exists")
    else:
        print("Starting from scratch")


# In[ ]:


for epoch in trange(start_epoch, nepoch, leave=False):
    print(epoch)
    lr_Linear(nepoch, epoch, lr)
    vae_net.train()
    for i, (images, _) in enumerate(tqdm(trainloader, leave=False)):

        recon_data, mu, logvar = vae_net(images.to(device))
        #VAE loss
        loss = vae_loss(recon_data, images.to(device), mu, logvar)        
        
        #Perception loss
        loss += feature_extractor(torch.cat((torch.sigmoid(recon_data), images.to(device)), 0))
    
        loss_log.append(loss.item())
        vae_net.zero_grad()
        loss.backward()
        optimizer.step()

    #In eval mode the model will use mu as the encoding instead of sampling from the distribution
    vae_net.eval()
    with torch.no_grad():
        recon_data, mu, logvar = vae_net(test_images.to(device))
        recon_data = recon_data.detach()
        loss = torch.add(mu.detach().sum(), logvar.detach().sum())
        #vutils.save_image(torch.cat((torch.sigmoid(recon_data.cpu()), test_images),2),"%s/%s/%s_%d.png" % (save_dir, "Results" , model_name, image_size))
        
        if not os.path.exists("%s/%s/%s/" % (save_dir, "Results" , model_name)):
            os.mkdir("%s/%s/%s/" % (save_dir, "Results" , model_name))
        if epoch % 2 == 0:
            vutils.save_image(torch.sigmoid(recon_data[0].cpu()),"%s/%s/%s/%s_%s_%s.png" % (save_dir, "Results" , model_name, 'recon', epoch, loss))
            vutils.save_image(test_images[0],"%s/%s/%s/%s_%s.png" % (save_dir, "Results" , model_name, 'original', epoch))
        

        #Save a checkpoint
        torch.save({
                    'epoch'                         : epoch,
                    'loss_log'                      : loss_log,
                    'model_state_dict'              : vae_net.state_dict(),
                    'optimizer_state_dict'          : optimizer.state_dict()

                        }, save_dir + "/Models/" + model_name + "_" + str(image_size) + ".pt")  


# %%
