import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import custom_bwd, custom_fwd

N_FEATURE = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ParticleDataset(Dataset):
    def __init__(self, features, distances_1, distances_0, distances_m1, masks, labels):
        self.features = features
        self.distances_1 = distances_1
        self.distances_0 = distances_0
        self.distances_m1 = distances_m1
        self.masks = masks
        self.labels = labels
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.Tensor(self.features[idx])
        distance_1 = torch.Tensor(self.distances_1[idx])
        distance_0 = torch.Tensor(self.distances_0[idx])
        distance_m1 = torch.Tensor(self.distances_m1[idx])
        mask = torch.Tensor(self.masks[idx])
        label = torch.Tensor(self.labels[idx])
        return feature, distance_1, distance_0, distance_m1, mask, label
        
        
class MPNN(nn.Module):
    def __init__(self, input_features): 
        super(MPNN, self).__init__() 
        self.input_features = input_features
        self.fe = nn.Linear(input_features, N_FEATURE) 
        self.fm1 = nn.Linear(N_FEATURE + 12, N_FEATURE)  
        self.fu1 = nn.Linear(N_FEATURE * 2, N_FEATURE) 
        self.fmm1 = nn.Linear(N_FEATURE + 12, N_FEATURE)    
        self.fum1 = nn.Linear(N_FEATURE * 2, N_FEATURE) 
        self.fm0 = nn.Linear(N_FEATURE + 12, N_FEATURE)   
        self.fu0 = nn.Linear(N_FEATURE * 2, N_FEATURE) 
        self.distance_nodes = torch.linspace(0, 0.3, 12).view(1, -1).cuda() 
    def forward(self, x, d1, d0, dm1, mask): 
    
        h = F.relu(self.fe(x))
        d1_first_element = d1[:, :, :, 0:1]
        d1_second_element = d1[:, :, :, 1]
        d1_expanded = torch.exp(-(d1_second_element.unsqueeze(-1) - self.distance_nodes)**2 / 2 / 0.015**2)
        d1_final = torch.cat([d1_first_element, d1_expanded], dim=-1)
        particles_indices1 = d1_final[:,:,:,0]
        particles_size1 = d1.size(2)
        d1 = d1_final[:,:,:,1:]
        
        dm1_first_element = dm1[:, :, :, 0:1]
        dm1_second_element = dm1[:, :, :, 1]
        dm1_expanded = torch.exp(-(dm1_second_element.unsqueeze(-1) - self.distance_nodes)**2 / 2 / 0.015**2)
        dm1_final = torch.cat([dm1_first_element, dm1_expanded], dim=-1)
        particles_indicesm1 = dm1_final[:,:,:,0]
        particles_sizem1 = dm1.size(2)
        dm1 = dm1_final[:,:,:,1:]
        
        d0_first_element = d0[:, :, :, 0:1]
        d0_second_element = d0[:, :, :, 1]
        d0_expanded = torch.exp(-(d0_second_element.unsqueeze(-1) - self.distance_nodes)**2 / 2 / 0.015**2)
        d0_final = torch.cat([d0_first_element, d0_expanded], dim=-1)
        particles_indices0 = d0_final[:,:,:,0]
        particles_size0 = d0.size(2)
        d0 = d0_final[:,:,:,1:]
        
        # message passing  
        hj1 = torch.gather(h.unsqueeze(2).expand(-1, -1, particles_size1, -1), 1, particles_indices1.unsqueeze(-1).expand(-1, -1, -1, N_FEATURE).long())
        hd1 = torch.cat((hj1, d1), dim=-1)  
        m1 = F.relu(self.fm1(hd1)).sum(dim=2)
        m1 = torch.cat([h, m1], dim=-1)
        h1 = F.sigmoid(self.fu1(m1))
        
        # message passing  
        hj0 = torch.gather(h.unsqueeze(2).expand(-1, -1, particles_size0, -1), 1, particles_indices0.unsqueeze(-1).expand(-1, -1, -1, N_FEATURE).long())
        hd0 = torch.cat((hj0, d0), dim=-1)  
        m0 = F.relu(self.fm0(hd0)).sum(dim=2)
        m0 = torch.cat([h, m0], dim=-1)
        h0 = F.sigmoid(self.fu0(m0))
        
        # message passing  
        hjm1 = torch.gather(h.unsqueeze(2).expand(-1, -1, particles_sizem1, -1), 1, particles_indicesm1.unsqueeze(-1).expand(-1, -1, -1, N_FEATURE).long())
        hdm1 = torch.cat((hjm1, dm1), dim=-1)  
        mm1 = F.relu(self.fmm1(hdm1)).sum(dim=2)
        mm1 = torch.cat([h, mm1], dim=-1)
        hm1 = F.sigmoid(self.fum1(mm1))
        return h1, h0, hm1        
        
        
class SSM(nn.Module):
    def __init__(self, D, N, L):
        super(SSM, self).__init__()
        self.D = D  # Dimension of each sequence
        self.N = N  # State size
        self.L = L  # Sequence length

        # Define the A matrix
        self.A = nn.Parameter(torch.full((D, N), -0.5))

        # Define the linear projection layer of B and C
        self.B_projection = nn.Linear(D, N)
        init.xavier_uniform_(self.B_projection.weight)
        
        self.C_projection = nn.Linear(D, N)
        init.xavier_uniform_(self.C_projection.weight)
        # Define the linear projection layer of Δ. First project to 1D and then broadcast
        self.delta_projection = nn.Linear(D, 1)
        init.uniform_(self.delta_projection.weight, 0.001, 0.1)

    def forward(self, x):
        B_list = []
        C_list = []
        ys = []
        # Apply the projection of B and C on every timestep        
           
        for t in range(self.L):
            x_t = x[:, :, t]  # Select the input at time step t, shape: (B, D)
            B_t = self.B_projection(x_t)  # (B, N)
            C_t = self.C_projection(x_t)  # (B, N)
            B_list.append(B_t)
            C_list.append(C_t)

        # Stack the projections along the time dimension
        B = torch.stack(B_list, dim=1)  # (B, L, N)
        C = torch.stack(C_list, dim=1)  # (B, L, N)
        # Apply delta projection for each time step
        delta_list = [self.delta_projection(x[:, :, t]).squeeze(-1) for t in range(self.L)]
        delta = torch.stack(delta_list, dim=1)  # (B, L)
        delta = delta.unsqueeze(-1).expand(-1, -1, self.D)  # Broadcast to (B, L, D)
        delta = F.softplus(delta)  
        I = torch.ones((x.size(0), 4037, 12, 12)).to(device)
        
        # discretize
        delta_A = torch.einsum('bld,dn->bldn', delta, self.A)
        Abar = torch.exp(delta_A)
        delta_A_inv = 1/(delta_A+1e-5)
        exp_delta_A_minus_I = Abar - I
        deltaB_x = delta_A_inv*exp_delta_A_minus_I*torch.einsum('bld,bln,bld->bldn', delta, B, x.permute(0, 2, 1))
        h = torch.sigmoid(torch.randn(x.size(0), self.D, self.N)).to(device)  # Normal distribution
        for i in range(1, 4038):    
            h = Abar[:, i-1, :, :].squeeze(1) * h + deltaB_x[:, i-1, :, :].squeeze(1)
            y = torch.einsum('bdn,bn->bd', h.squeeze(1), C[:, i-1, :]) if i > 0 else torch.zeros(16, self.D) 
            ys.append(y)
        yp = torch.stack(ys, dim=2) # (B, D, L)
        
        return yp

class MambaModel(nn.Module):
    def __init__(self, input_features, expanded_features, kernel_size, particle_number):
        super(MambaModel, self).__init__()
        self.expand_linear1 = nn.Linear(input_features, expanded_features)
        self.expand_linear2 = nn.Linear(input_features, expanded_features)
        self.kernel_size = kernel_size
        self.particle_number = particle_number
        self.expanded_features = expanded_features
        self.output_projection = nn.Linear(expanded_features, 1)
        self.conv1d_weight = nn.Parameter(torch.randn(expanded_features, expanded_features, kernel_size))
        self.conv1d_bias = nn.Parameter(torch.randn(expanded_features))
        self.SSM = SSM(12, 12, 4037).to(device)
   
    def forward(self, x):       
        expanded1 = F.silu(self.expand_linear1(x))
        expanded2 = F.silu(self.expand_linear2(x))
        #shape transformation [batch_size, expanded_features, num_particles]
        expanded1 = expanded1.permute(0, 2, 1)
        expanded2 = expanded2.permute(0, 2, 1)
        expanded1 = F.conv1d(expanded1, self.conv1d_weight, self.conv1d_bias, padding=self.kernel_size-1, groups=1)
        expanded1 = expanded1[:, :, :self.particle_number]
        expanded1 = F.silu(expanded1)
        expanded1 = self.SSM(expanded1)
        
        # Element-wise multiplicatioon
        combined = expanded1 * expanded2  #[batch_size, 32, num_particles]
        combined = combined.permute(0, 2, 1)
        
        return combined
                
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba1 = MambaModel(input_features=6, expanded_features=12, kernel_size = 3, particle_number = 4037).to(device)
        self.mamba2 = MambaModel(input_features=12, expanded_features=12, kernel_size = 3, particle_number = 4037).to(device)
        self.mamba3 = MambaModel(input_features=12, expanded_features=12, kernel_size = 3, particle_number = 4037).to(device)
        self.mpnn1 = MPNN(6)
        self.mpnn2 = MPNN(12)
        self.mpnn3 = MPNN(12)
        self.fc1 = nn.Linear(36, 12)  
        self.fc2 = nn.Linear(48, 12)
        self.fc3 = nn.Linear(48, 12)
        self.output_projection = nn.Linear(12, 1)
    def forward(self, x, d1, d0, dm1, mask):
        #loop 1
        x1, x2, x3 = self.mpnn1(x, d1, d0, dm1, mask)
        #x4 = self.mamba1(x)
        concatenated_output = torch.cat([x1,x2,x3],dim = -1)
        x = F.relu(self.fc1(concatenated_output))
        #loop 2
        x1, x2, x3 = self.mpnn2(x, d1, d0, dm1, mask)
        x4 = self.mamba2(x)
        concatenated_output = torch.cat([x1,x2,x3,x4],dim = -1)
        x = F.relu(self.fc2(concatenated_output))
        #loop 3
        x1, x2, x3 = self.mpnn3(x, d1, d0, dm1, mask)
        x4 = self.mamba3(x)
        concatenated_output = torch.cat([x1,x2,x3,x4],dim = -1)
        x = F.relu(self.fc3(concatenated_output))
        combined_reshaped = x.reshape(-1, x.size(2))  #[batch_size * num_particles, 12]
        combined_transformed = self.output_projection(combined_reshaped)  #[batch_size * num_particles, 1]
        combined_transformed = combined_transformed.view(-1, 4037, 1)  #[batch_size, num_particles, 1]
        output = torch.sigmoid(combined_transformed)  # [batch_size, num_particles, 1]
        output = output.squeeze(1)  # [batch_size, num_particles]
        return output
        
model = MyModel().to(device)
for name, p in model.named_parameters():
    if name.endswith('.weight'):
        torch.nn.init.xavier_normal_(p)
    elif name.endswith('.bias'):
        torch.nn.init.constant_(p, 0)

train_features_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/train{i}.npy'
    train_features_path.append(file_path)

valid_features_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/valid{i}.npy'
    valid_features_path.append(file_path)

train_masks_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/train_mask{i}.npy'
    train_masks_path.append(file_path)

valid_masks_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/valid_mask{i}.npy'
    valid_masks_path.append(file_path)
    
train_R1s_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/train_R1_{i}.npy'
    train_R1s_path.append(file_path)

valid_R1s_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/valid_R1_{i}.npy'
    valid_R1s_path.append(file_path)
    
train_R0s_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/train_R0_{i}.npy'
    train_R0s_path.append(file_path)

valid_R0s_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/valid_R0_{i}.npy'
    valid_R0s_path.append(file_path)

train_Rm1s_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/train_Rm1_{i}.npy'
    train_Rm1s_path.append(file_path)

valid_Rm1s_path = []
for i in range(1):
    file_path = f'/home/daohan/apps/mamba/sample/valid_Rm1_{i}.npy'
    valid_Rm1s_path.append(file_path)

criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
for epoch in range(num_epochs):
    print('\033[1;32m', 'Epoch', epoch, '\033[0m')
    model.train()
    train_loss = 0.0
    valid_loss = 0.0
    for i in range(1):
        train_features = np.load(train_features_path[i])[:, :, :-1]
        train_distances_1 = np.load(train_R1s_path[i]) 
        train_distances_0 = np.load(train_R0s_path[i]) 
        train_distances_m1 = np.load(train_Rm1s_path[i]) 
        train_masks = np.load(train_masks_path[i]) 
        train_labels = np.load(train_features_path[i])[:, :, -1]
        train_dataset = ParticleDataset(train_features,train_distances_1,train_distances_0,train_distances_m1,train_masks,train_labels)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        for train_feature, train_distance_1, train_distance_0, train_distance_m1, train_mask, train_label in tqdm(train_loader, total=len(train_loader)):  
            train_feature = train_feature.to(device)
            train_distance_1 = train_distance_1.to(device)
            train_distance_0 = train_distance_0.to(device)
            train_distance_m1 = train_distance_m1.to(device)
            train_mask = train_mask.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            output = model(train_feature, train_distance_1, train_distance_0, train_distance_m1, train_mask).squeeze()
            ptmax = train_feature[:, :, 0].max(dim=1, keepdim=True)[0]    
            weight = train_feature[:, :, 0]/ptmax
            per_particle_loss = criterion(output, train_label) 
            loss = (per_particle_loss * weight).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del train_feature
            del train_distance_1
            del train_distance_0
            del train_distance_m1
            del train_mask
            del train_label
            del weight
            torch.cuda.empty_cache()
        del train_features
        del train_distances_1
        del train_distances_0
        del train_distances_m1
        del train_masks
        del train_labels
        torch.cuda.empty_cache()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f'Average Training Loss: {avg_train_loss:.4f}')
    
    torch.save(model.state_dict(), '/home/daohan/apps/mamba/weights/model-%.3d.weights' % epoch)   
    
    for j in range(1):
        valid_features = np.load(valid_features_path[j])[:, :, :-1] 
        valid_distances_1 = np.load(valid_R1s_path[j]) 
        valid_distances_0 = np.load(valid_R0s_path[j]) 
        valid_distances_m1 = np.load(valid_Rm1s_path[j]) 
        valid_masks = np.load(valid_masks_path[j]) 
        valid_labels = np.load(valid_features_path[j])[:, :, -1] 
        valid_dataset = ParticleDataset(valid_features,valid_distances_1,valid_distances_0,valid_distances_m1,valid_masks,valid_labels)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True) 
        model.eval() 
        with torch.no_grad(): 
            for valid_feature, valid_distance_1, valid_distance_0, valid_distance_m1, valid_mask, valid_label in tqdm(valid_loader, total=len(valid_loader)):  
                valid_feature = valid_feature.to(device)
                valid_distance_1 = valid_distance_1.to(device)
                valid_distance_0 = valid_distance_0.to(device)
                valid_distance_m1 = valid_distance_m1.to(device)
                valid_mask = valid_mask.to(device)
                valid_label = valid_label.to(device)
                optimizer.zero_grad()
                output = model(valid_feature, valid_distance_1, valid_distance_0, valid_distance_m1, valid_mask).squeeze()
                ptmax = valid_feature[:, :, 0].max(dim=1, keepdim=True)[0]    
                weight = valid_feature[:, :, 0]/ptmax
                per_particle_loss = criterion(output, valid_label) 
                loss = (per_particle_loss * weight).sum(dim=1).mean()
                valid_loss += loss.item()
                del valid_feature
                del valid_distance_1
                del valid_distance_0
                del valid_distance_m1
                del valid_mask
                del valid_label
                del weight
                torch.cuda.empty_cache()
        del valid_features
        del valid_distances_1
        del valid_distances_0
        del valid_distances_m1
        del valid_masks
        del valid_labels
        torch.cuda.empty_cache()
    
    avg_valid_loss = valid_loss / len(valid_loader)
    print(f'Average Valid Loss: {avg_valid_loss:.4f}')
    with open('/home/daohan/apps/mamba/results.txt', 'a') as f:
        f.write(f'{epoch} {avg_train_loss} {avg_valid_loss}\n')

