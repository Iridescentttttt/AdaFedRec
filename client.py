import random
import torch.nn.functional as F
import numpy as np
import torch
from ncf_client import NCF_client
from ncf_client_low2 import NCF_client_low2
from ncf_client_low4 import NCF_client_low4
from metrics import MetronAtK
import copy
import torch.optim as optim


class Client:
    def __init__(self, config):
        self.config = config

        # Randomly select a model type
        model_types = ['NCF_client', 'NCF_client_low2', 'NCF_client_low4']

        self.selected_model_type = ""
        if self.config['random_client'] == True:
            self.selected_model_type = random.choice(model_types)
            # Initialize the selected model
            if self.selected_model_type == 'NCF_client':
                self.model = NCF_client(config)
            elif self.selected_model_type == 'NCF_client_low2':
                self.model = NCF_client_low2(config)
            elif self.selected_model_type == 'NCF_client_low4':
                self.model = NCF_client_low4(config)
            else:
                raise ValueError(f"Unknown model type: {self.selected_model_type}")
        else:
            ratios = self.config['model_type_ratios']

            # Calculate the cumulative probabilities
            cumulative_ratios = [sum(ratios[:i + 1]) for i in range(len(ratios))]
            self.selected_model_type = 'NCF_client'
            # Randomly select a model type based on the specified ratios
            rand_val = random.uniform(0, cumulative_ratios[-1])
            for i, cum_ratio in enumerate(cumulative_ratios):
                if rand_val <= cum_ratio:
                    self.selected_model_type = model_types[i]
                    break
            # Initialize the selected model
            if self.selected_model_type == 'NCF_client':
                self.model = NCF_client(config)
            elif self.selected_model_type == 'NCF_client_low2':
                self.model = NCF_client_low2(config)
            elif self.selected_model_type == 'NCF_client_low4':
                self.model = NCF_client_low4(config)
            else:
                raise ValueError(f"Unknown model type: {self.selected_model_type}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.device = torch.device(f"cuda:{self.config['device_id']}" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.crit = torch.nn.BCELoss()
        self._metron = MetronAtK(top_k=self.config['top_k'])

    def local_train(self, dataloader):
        """
        Perform local training on a user's private data.
        """
        self.model.to(self.device)
        self.model.train()
        all_loss = 0.0

        # Optional: save initial model parameters for tracking parameter changes
        # initial_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

        for epoch in range(self.config['local_epoch']):
            for batch_id, batch in enumerate(dataloader):
                # Ensure input is of correct type
                assert isinstance(batch[0], torch.LongTensor)

                users, items, ratings = batch[0], batch[1], batch[2]
                ratings = ratings.float()
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)

                self.optimizer.zero_grad()
                ratings_pred = self.model(users, items)
                loss = self.crit(ratings_pred.view(-1), ratings)
                all_loss += loss.item()

                loss.backward()
                self.optimizer.step()

        # Optional: move model back to CPU to save memory
        # self.model.to(torch.device("cpu"))

        # Optional: track parameter change for analysis or debugging
        # final_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        # param_changes = {name: final_params[name] - initial_params[name] for name in initial_params}

        return all_loss

    def evaluate(self, test_users, test_items, negative_users, negative_items):
        """
        Evaluate the model using test and negative samples.

        Args:
            test_users (Tensor): User IDs corresponding to positive samples.
            test_items (Tensor): Item IDs for the positive interactions.
            negative_users (Tensor): User IDs for negative samples.
            negative_items (Tensor): Item IDs for negative (non-interacted) items.

        Returns:
            tuple: (recall, ndcg), both float values.
        """
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            # Move data to the appropriate device
            test_users = test_users.to(self.device)
            test_items = test_items.to(self.device)
            negative_users = negative_users.to(self.device)
            negative_items = negative_items.to(self.device)

            # Match user-item pair dimensions if needed (broadcasting users)
            if test_users.size(0) != test_items.size(0):
                test_users = test_users.repeat(test_items.size(0))

            if negative_users.size(0) != negative_items.size(0):
                negative_users = negative_users.repeat(negative_items.size(0))

            # Compute prediction scores for positive and negative samples
            test_scores = self.model(test_users, test_items)
            negative_scores = self.model(negative_users, negative_items)

            # Move data back to CPU for evaluation
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()

            # Set evaluation data for the metric calculator
            self._metron.subjects = [
                test_users.data.view(-1).tolist(),
                test_items.data.view(-1).tolist(),
                test_scores.data.view(-1).tolist(),
                negative_users.data.view(-1).tolist(),
                negative_items.data.view(-1).tolist(),
                negative_scores.data.view(-1).tolist()
            ]

            # Compute Recall@K and NDCG@K
            recall = self._metron.cal_recall()
            ndcg = self._metron.cal_ndcg()

            # Optional: release GPU memory
            # self.model.to(torch.device("cpu"))

            return recall, ndcg

    def change_model(self, target_model, dataloader, temperature, alpha, server_model):
        assert target_model != self.model.__class__.__name__, "Target model must be different from the current model"

        # Save the current model state
        teacher_model = copy.deepcopy(self.model)
        teacher_state_dict = teacher_model.state_dict()

        # Initialize the target model
        self.selected_model_type = target_model
        if target_model == 'NCF_client':
            self.model = NCF_client(self.config)
        elif target_model == 'NCF_client_low2':
            self.model = NCF_client_low2(self.config)
        elif target_model == 'NCF_client_low4':
            self.model = NCF_client_low4(self.config)
        else:
            raise ValueError(f"Unknown target model: {target_model}")

        # Load the common layers' parameters from the original model
        student_state_dict = self.model.state_dict()
        for name, param in teacher_state_dict.items():
            if name not in ['embedding_user.weight', 'embedding_item.weight', 'mlp.0.weight']:
                student_state_dict[name].copy_(param)

        # Update the item_embedding and mlp.0.weight using the server_model's autoencoders
        if target_model == 'NCF_client':
            student_state_dict['embedding_item.weight'] = server_model.state_dict()['embedding_item.weight']
            student_state_dict['mlp.0.weight'] = server_model.state_dict()['mlp.0.weight']
        elif target_model == 'NCF_client_low2':
            student_state_dict['embedding_item.weight'] = server_model.autoencoder_item_low2.encoder(
                server_model.state_dict()['embedding_item.weight'])
            student_state_dict['mlp.0.weight'] = server_model.autoencoder_mlp_first_low2.encoder(
                server_model.state_dict()['mlp.0.weight'])
        elif target_model == 'NCF_client_low4':
            student_state_dict['embedding_item.weight'] = server_model.autoencoder_item_low4.encoder(
                server_model.state_dict()['embedding_item.weight'])
            student_state_dict['mlp.0.weight'] = server_model.autoencoder_mlp_first_low4.encoder(
                server_model.state_dict()['mlp.0.weight'])

        self.model.load_state_dict(student_state_dict)
        # Freeze the specified layers
        for name, param in self.model.named_parameters():
            if name not in ['embedding_user.weight', 'embedding_item.weight', 'mlp.0.weight']:
                param.requires_grad = False

        # optimizer for the unfrozen layers
        # Because embedding_item and model.mlp[0] contains semantic information, so the lr is lower than embedding_user
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.optimizer = optim.Adam([
            {'params': self.model.embedding_user.parameters(), 'lr': 0.01},
            {'params': self.model.embedding_item.parameters(), 'lr': 0.001},
            {'params': self.model.mlp[0].parameters(), 'lr': 0.001}
        ])

        # knowledge distillation
        self.model.to(self.device)
        teacher_model.to(self.device)
        total_loss = 0
        epochs = self.config['KD_epoch']

        for epoch in range(epochs):
            self.model.train()
            teacher_model.eval()
            all_loss = 0.0
            for batch_id, batch in enumerate(dataloader):
                users, items, ratings = batch[0], batch[1], batch[2]
                ratings = ratings.float()
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                self.optimizer.zero_grad()

                # Get predictions from student model
                student_preds = self.model(users, items)
                student_mlp_output = self.model.mlp[0](
                    torch.cat([self.model.embedding_user(users), self.model.embedding_item(items)], dim=-1))

                # Get predictions from teacher model
                with torch.no_grad():
                    teacher_preds = teacher_model(users, items)
                    teacher_mlp_output = teacher_model.mlp[0](
                        torch.cat([teacher_model.embedding_user(users), teacher_model.embedding_item(items)], dim=-1))

                # Compute the knowledge distillation loss
                loss_kd = F.kl_div(F.log_softmax(student_preds / temperature, dim=1),
                                   F.softmax(teacher_preds / temperature, dim=1),
                                   reduction='batchmean') * (temperature ** 2)

                # Compute the standard loss
                loss_standard = self.crit(student_preds.view(-1), ratings)

                loss = alpha * loss_standard + (1 - alpha) * loss_kd

                all_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            # print(f'Epoch {epoch + 1}/{epochs}, Loss: {all_loss / len(dataloader)}')
            total_loss = all_loss / len(dataloader)

        # For the following training, restore to original training optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if name not in ['embedding_user.weight', 'embedding_item.weight', 'mlp.0.weight']:
                param.requires_grad = True
        return total_loss
