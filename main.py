#%%
from matplotlib import projections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url

#%%
import os
import random
import numpy as np
import math
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange, tqdm
import matplotlib as mpl
from model.RES_VAE import VAE as VAE
from model.vgg19 import VGG19

from data import get_data

use_cuda = torch.cuda.is_available()
GPU_indx  = 0
device = torch.device(GPU_indx if use_cuda else "cpu")

#%%


from clearml import StorageManager, Task, OutputModel, Logger
from clearml import Dataset as cmlDataset
task = Task.init(project_name='bogdoll/anomaly_detection_simon', task_name='cnn-vae')
logger = task.get_logger
#model = OutputModel(task=task)

task.set_base_docker(
            "nvcr.io/nvidia/pytorch:21.10-py3",
            docker_setup_bash_script="apt-get update && apt-get install -y libfreetype6-dev && apt-cache search freetype | grep dev && apt-get install -y python3-opencv",
            docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all"
            )
task.execute_remotely('docker', clone=False, exit_process=True)



batch_size = 8
img_size = 256
lr = 1e-4
num_epochs = 100
#dataset_root = "/disk/vanishing_data/mb274/data/cityscapes/test/"
dataset_root = cmlDataset.get(dataset_name= 'cityscapes_train', dataset_project= 'bogdoll/anomaly_detection_simon').get_local_copy()
save_dir = os.getcwd()
model_name = "vae"
load_checkpoint  = False


#data = {"normal": {"0": '/disk/vanishing_data/mb274/mnist_dummy/normal/'}, "anomaly": {"1": '/disk/vanishing_data/mb274/mnist_dummy/anomaly/'}, "anomaly_test": None}

data = {"normal": {"0": dataset_root}, "anomaly": {"1": dataset_root}, "anomaly_test": None}

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
    img_cat = torch.cat((img, recon_data), 0)
    out = feature_extractor(img_cat)
    loss = 0
    for i in range(len(feature_extractor)):
        if isinstance(feature_extractor[i], GetFeatures):
            loss += (feature_extractor[i].features[:(img.shape[0])] - feature_extractor[i].features[(img.shape[0]):]).pow(2).mean()
    return loss/(i+1)

#Linear scaling the learning rate down
def lr_Linear(epoch_max, epoch, lr):
    lr_adj = ((epoch_max-epoch)/epoch_max)*lr
    set_lr(lr = lr_adj)

def set_lr(lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def get_vae_loss_1(log_var, z_prior_mean, y, z, x):
    #print(log_var.shape)
    #log_var = log_var.squeeze(3)
    #print(log_var.shape, z_prior_mean.shape, y.shape)
    recon_loss = F.binary_cross_entropy_with_logits(recon, x)
    log_var = torch.unsqueeze(log_var, 1)
    kl_loss = - 0.5 * (log_var - torch.square(torch.unsqueeze(z, 1)- z_prior_mean))

    kl_loss = torch.mean(torch.bmm(torch.unsqueeze(y, 1).float(), kl_loss), 0)
    return torch.sum(kl_loss) + torch.sum(recon_loss)

def get_vae_loss_2(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy_with_logits(recon, x)
    KL_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    loss = recon_loss + 0.01 * KL_loss
    return loss
#%%
train_loader, val_loader, test_loader = get_data(data, img_size=img_size, batch_size=batch_size)
#train_loader, test_loader = get_data_STL10(batch_size=batch_size, root=dataset_root)
dataiter = iter(test_loader)
test_images, test_label, name = dataiter.next()

#%%
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

#Create VAE network
vae_net = VAE(channel_in=3, ch=256).to(device)

# setup optimizer
optimizer = optim.Adam(vae_net.parameters(), lr=0.0001)

#Loss function
#BCE_Loss = nn.BCEWithLogitsLoss()
#loss_log = []

#Create the save directory if it does note exist
if not os.path.isdir(save_dir + "/trained_models"):
    os.makedirs(save_dir + "/trained_models")
if not os.path.isdir(save_dir + "/Results"):
    os.makedirs(save_dir + "/Results")

if load_checkpoint:
    model_path = torch.load(save_dir + "/trained_models/" + model_name + "_" + str(img_size) + ".pt")
    vae_net.load_state_dict(model_path)
    print("Checkpoint loaded")
else:
    #If checkpoint does exist raise an error to prevent accidental overwriting
    if os.path.isfile(save_dir + "/trained_models/" + model_name + "_" + str(img_size) + ".pt"):
        raise ValueError("Warning Checkpoint exists")
    else:
        print("Starting with new model")

#%%
best_loss = math.inf
def to_categorical(y, num_classes):
    y_in = []
    for i in range(len(y)):
        y_in.append(np.eye(num_classes, dtype='float32')[int(y[i].item())])
    return torch.tensor(np.array(y_in))

for epoch in range(num_epochs):
    #lr_Linear(num_epochs, epoch, lr)
    train_loss = 0.0
    feat_loss = 0.0
    #vae_loss = 0.0
    val_loss = 0.0
    vae_net.train()
    with tqdm(train_loader, unit='batch') as tepoch:
        for data in tepoch:
            tepoch.set_description(f'train epoch {epoch+1}')
            img, label, name = data
            label = to_categorical(label, num_classes=2).to(device)
            label = label.to(device)
            optimizer.zero_grad()

            recon, z, mu, log_var, prior_mean = vae_net(img.to(device))
            
            #log_var = log_var.squeeze(-1).squeeze(-1).squeeze(1)
            #mu = mu.squeeze(-1).squeeze(-1).squeeze(1)
            #z = z.squeeze(-1).squeeze(-1)

            #loss = get_vae_loss_1(log_var, prior_mean, label, z, img.to(device))
            loss = get_vae_loss_2(recon, img.to(device), mu, log_var)

            
            feat_loss = feature_extractor(torch.cat((recon, img.to(device)), 0))
            loss = loss + feat_loss
            loss.backward()
            optimizer.step()
            
            #train_loss += loss.item()
            #feat_loss += feat_loss.item()
            #vae_loss += vae_loss.item()
            
        #print(f'vae_loss: {vae_loss} feature_loss: {feat_loss} train_loss: {train_loss}')
    '''
    vae_net.eval()
    with torch.no_grad():
        with tqdm(val_loader, unit='batch') as tepoch:
            for data in tepoch:
                tepoch.set_description(f'val epoch {epoch+1}')
                img, label, name = data
                #label = to_categorical(label, num_classes)
                #label = label.to(device)        
        
                recon_data, z, mu, logvar, prior_mean = vae_net(img.to(device))
                recon_data = recon_data.detach()
                loss = get_vae_loss(recon_data.cpu(), img.cpu(), mu.cpu(), logvar.cpu(), prior_mean.cpu(), evall=True)
                val_loss += loss.item()

            print(f'epoch: {epoch + 1} val_loss: {val_loss}')

    if val_loss < best_loss:
        torch.save(vae_net.state_dict(), save_dir + "/trained_models/" + model_name + "_" + str(img_size) + ".pt")
        best_loss = val_loss
        print('save model with lowest loss')
    '''
    if epoch % 5 == 0:
        recon_data, _, _, _, _ = vae_net(test_images.to(device))
        np_recon = np.moveaxis(np.array(recon_data[0].detach().cpu()), 0, 2)
        pil_recon = Image.fromarray(np.uint8(np_recon*255))
        np_original = np.moveaxis(np.array(test_images[0]), 0, 2)
        pil_original = Image.fromarray(np.uint8(np_original*255))
        pil_recon.save('recon_'+str(epoch)+'.png')
        pil_original.save('original_'+str(epoch)+'.png')
        task.upload_artifact("%s_%s.png" % ('recon', epoch), pil_recon)
        task.upload_artifact("%s_%s.png" % ('original', epoch), pil_original)


#%%

#plot_ls(vae_net, train_loader, test_loader, latent_dim=2, plot_dim=2)

        
#show_rand_recon(vae_net, test_loader)
# %%
