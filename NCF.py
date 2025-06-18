import torch
from engine import Engine

class NCF(torch.nn.Module):
    def __init__(self, config):
        super(NCF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_dim*2, out_features=8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=8),
            torch.nn.ReLU(),
        )
        self.affine_output = torch.nn.Linear(in_features=8, out_features=1)
        self.logistic = torch.nn.Sigmoid()
        self.init_weight()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        vector = self.mlp(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        torch.nn.init.normal_(self.embedding_item.weight, std=0.01)
        torch.nn.init.normal_(self.embedding_user.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
        torch.nn.init.kaiming_uniform_(self.affine_output.weight, a=1)


class NCFEngine(Engine):
    """Engine for training & evaluating NCF model"""
    def __init__(self, config):
        self.model = NCF(config)
        if config['use_cuda'] is True:
            self.model.cuda()
        super(NCFEngine, self).__init__(config)
        print(self.model)
