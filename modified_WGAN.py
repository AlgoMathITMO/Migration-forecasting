import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import copy
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset


from sklearn.metrics import mean_absolute_percentage_error, make_scorer, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier, XGBRFRegressor
import optuna
from functools import partial

from rdt.transformers.numerical import GaussianNormalizer
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

torch.manual_seed(0)


#############################################################################
# MODULES
#############################################################################

class RegNN(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(RegNN, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1))

    def forward(self, x):
        out = self.fc(x)
        
        return out
    
    
class Generator(nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(Generator, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                # nn.ReLU(),
                                # nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out = self.fc(x)
        
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(Discriminator, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                # nn.ReLU(),
                                # nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1))

    def forward(self, x):
        out = self.fc(x)
        
        return out
    
    
class training_set(Dataset):
    def __init__(self,X,Y):
        self.X = X                        
        self.Y = Y                         

    def __len__(self):
        return len(self.X)                

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]] 
    

#############################################################################





#############################################################################
# NEURAL NETWORK FOR REGRESSION
#############################################################################

def train_reg_NN(synth_data, batch_size=64, epochs=2000):
    # create dataloader
    X_dl = DataLoader(training_set(torch.FloatTensor(synth_data.drop(['saldo'], axis=1).values).to(torch.float32),
                    torch.FloatTensor(synth_data['saldo'].values).to(torch.float32)), batch_size=batch_size, shuffle=True)
    
    
    # initialize model and optimizator
    reg_NN = RegNN(2**8, len(synth_data.columns) - 1)
    criterion = nn.MSELoss()
    optimizer_reg = torch.optim.Adam(reg_NN.parameters(), lr=3e-4)
        
        
    reg_losses = []

    for _ in tqdm(range(epochs)):
        for X in X_dl:
            pred = reg_NN(X[0])
            loss = criterion(X[1].unsqueeze(1), pred)

            optimizer_reg.zero_grad()
            loss.backward()
            optimizer_reg.step()
        
        reg_losses.append(loss.item())
        
    
    return reg_NN, reg_losses


#############################################################################



#############################################################################
# MODIFIED WGAN
#############################################################################

def gradient_penalty(real_data, generated_data, discriminator):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)


        # Calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs= torch.ones(prob_interpolated.size()),
                            create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return 10 * ((gradients_norm - 1) ** 2).mean()



def train_modified_WGAN(synth_data, reg_NN, epochs=2000):
    synth_data_tr = copy.deepcopy(synth_data)
    transformer_arr = []

    # normalize data
    for col in synth_data.columns:
        transformer = GaussianNormalizer()
        synth_data_tr[col] = transformer.fit_transform(synth_data, column=[col])[col]
        transformer_arr.append(transformer)
        
    # initialize constants
    batch_size = 128
    noise_dim = 10
    input_size = len(synth_data.columns)
    
    # initialize models
    generator = Generator(2**7, noise_dim + input_size, input_size)
    discriminator = Discriminator(2**7, input_size)
    
    # create dataloader
    D_dl = DataLoader(torch.FloatTensor(synth_data_tr.values), batch_size=batch_size, shuffle=True)
        
    # initialize optimizators
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=3e-4)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=3e-4)
    
    
    loss_mse = nn.MSELoss()
    losses = {'gen_loss': [], 'dis_loss': []}

    for _ in tqdm(range(epochs)):
        for batch_idx, D in enumerate(D_dl):
            noise = torch.randn((len(D), noise_dim))
            D_tilde = generator(torch.cat([noise, D], dim=1))
            # D_tilde = generator(noise)
        

            discriminator.trainable = True
            dis_loss = (-torch.mean(discriminator(D)) + torch.mean(discriminator(D_tilde))) + gradient_penalty(D, D_tilde, discriminator)


            optimizer_dis.zero_grad(set_to_none=False)
            dis_loss.backward()
            optimizer_dis.step()
                            
            if batch_idx % 5 == 0:
                
                discriminator.trainable = False
                x = generator(torch.cat([noise, D], dim=1))
                # x = generator(noise)
                gen_loss = -torch.mean(discriminator(x)) + loss_mse(reg_NN(x[:, :-1]).view(-1, 1), D[:, -1].view(-1, 1))
                # + loss_mse(reg_NN(x[:, :-1]).view(-1, 1), D[:, -1].view(-1, 1)) + loss_mse(x[:, -1].view(-1, 1), D[:, -1].view(-1, 1))
                
                optimizer_gen.zero_grad(set_to_none=False)
                gen_loss.backward()
                optimizer_gen.step()
                
        losses['gen_loss'].append(gen_loss.item())
        losses['dis_loss'].append(dis_loss.item())
        
        
    synth_enh = generator(torch.cat([torch.randn((len(synth_data_tr), noise_dim)),
                                     torch.FloatTensor(synth_data_tr.values[:,:])], dim=1)).detach().numpy()
    synth_enh = pd.DataFrame(synth_enh, columns=synth_data.columns)
    
    for i, col in enumerate(synth_enh.columns):
        synth_enh[col] = transformer_arr[i].reverse_transform(synth_enh)[col]


    return synth_enh, losses

#############################################################################
        
        
