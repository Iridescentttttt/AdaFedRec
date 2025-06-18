import torch
from AutoEncoder import Autoencoder
import torch.nn as nn
import torch.optim as optim
class NCF_server(torch.nn.Module):
    def __init__(self, config):
        super(NCF_server, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_dim*2, out_features=8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=8),
            torch.nn.ReLU(),
        )
        self.affine_output = torch.nn.Linear(in_features=8, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # Autoencoder1, for item embedding of medium models
        self.autoencoder_item_low2 = Autoencoder(self.latent_dim, self.latent_dim // 2)
        # Autoencoder2, for adapter of medium models
        self.autoencoder_mlp_first_low2 = Autoencoder(self.latent_dim * 2, self.latent_dim)
        # Autoencoder3, for item embedding of small models
        self.autoencoder_item_low4 = Autoencoder(self.latent_dim, self.latent_dim // 4)
        # Autoencoder4, for adapter of small models
        self.autoencoder_mlp_first_low4 = Autoencoder(self.latent_dim * 2, self.latent_dim//2)

        self.init_weight()

    def train_autoencoders(self):
        autoencoders = [
            (self.autoencoder_item_low2, self.embedding_item.weight),
            (self.autoencoder_mlp_first_low2, self.mlp[0].weight),
            (self.autoencoder_item_low4, self.embedding_item.weight),
            (self.autoencoder_mlp_first_low4, self.mlp[0].weight)
        ]

        criterion = nn.MSELoss()
        for autoencoder, data in autoencoders:
            optimizer = optim.Adam(autoencoder.parameters(), lr=self.config['autoencoder_lr'])
            for epoch in range(self.config['autoencoder_train_epoch']):
                optimizer.zero_grad()
                outputs = autoencoder(data)
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
                # if (epoch+1) % 10 == 0:
                #     print(f'Autoencoder training epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    def init_weight(self):
        torch.nn.init.normal_(self.embedding_item.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
        torch.nn.init.kaiming_uniform_(self.affine_output.weight, a=1)
