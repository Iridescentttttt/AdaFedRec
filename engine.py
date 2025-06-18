import copy
import logging
import time

from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from ncf_server import NCF_server
import random
import torch
from client import Client
from metrics import MetronAtK
from ncf_client import NCF_client
from ncf_client_low2 import NCF_client_low2
from ncf_client_low4 import NCF_client_low4
from utils import save_checkpoint


class Engine(object):
    def __init__(self, config):
        self.config = config
        self.server_model_param = {}
        self.client_model_params = {}
        self.device = torch.device(f"cuda:{self.config['device_id']}" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.server_model = NCF_server(config)
        self.clients = self.generate_clients()

    def generate_clients(self):
        start_index = 0
        clients = []
        for i in range(self.config['num_users']):
            clients.append(Client(config=self.config))
            start_index += 1
        return clients

    def instance_user_train_loader(self, user_train_data):
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_a_round(self, all_train_data, test_data, round_id):
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = random.sample(range(self.config['num_users']), num_participants)
        else:
            participants = random.sample(range(self.config['num_users']), self.config['clients_sample_num'])

        self.server_model = self.server_model.to(self.device)

        # 测量 train_autoencoders 的执行时间
        start_time_autoencoders = time.time()
        self.server_model.train_autoencoders()
        end_time_autoencoders = time.time()
        autoencoders_time = end_time_autoencoders - start_time_autoencoders
        logging.info(f"train_autoencoders execute time: {autoencoders_time} 秒")

        all_loss = 0
        user_param_changes = {}

        # Process: Download, Test, Local Training, Federate
        # Download
        for user in participants:
            client = self.clients[user]
            client.model.to(self.device)
            # 3 kinds of models
            # For medium and small clients, layers from 'mlp.0.bias' to the last are taken into model2_dict
            # Then 'embedding_item.weight' and 'mlp.0.weight' are encoded
            pretext_model = self.server_model.state_dict()
            model2_dict = client.model.state_dict()
            if isinstance(client.model, NCF_client):
                state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
                model2_dict.update(state_dict)
            elif isinstance(client.model, NCF_client_low2):
                state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
                state_dict['embedding_item.weight'] = self.server_model.autoencoder_item_low2.encoder(
                    state_dict['embedding_item.weight'])
                state_dict['mlp.0.weight'] = self.server_model.autoencoder_mlp_first_low2.encoder(
                    state_dict['mlp.0.weight'])
                model2_dict.update(state_dict)
            elif isinstance(client.model, NCF_client_low4):
                state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
                state_dict['embedding_item.weight'] = self.server_model.autoencoder_item_low4.encoder(
                    state_dict['embedding_item.weight'])
                state_dict['mlp.0.weight'] = self.server_model.autoencoder_mlp_first_low4.encoder(
                    state_dict['mlp.0.weight'])
                model2_dict.update(state_dict)
            else:
                raise ValueError("Unknown client model type")
            client.model.load_state_dict(model2_dict)

        # Test
        test_recall = 0
        test_ndcg = 0

        large = 0
        medium = 0
        small = 0

        test_recall_large = 0
        test_recall_medium = 0
        test_recall_small = 0
        test_ndcg_large = 0
        test_ndcg_medium = 0
        test_ndcg_small = 0

        if round_id % self.config['test_per_epoch'] == 0:
            test_users, test_items = test_data[0], test_data[1]
            test_negative_users, test_negative_items = test_data[2], test_data[3]

            # If use gpu
            if self.config['use_cuda']:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                test_negative_users = test_negative_users.cuda()
                test_negative_items = test_negative_items.cuda()

            for user in participants:
                user_model = self.clients[user]
                with torch.no_grad():
                    # Obtain the positive samples of the current user
                    # (the positive samples related to the current user in test_items)
                    user_mask = test_users == user
                    test_item = test_items[user_mask]

                    # Obtain the negative samples of the current user.
                    neg_mask = test_negative_users == user
                    test_negative_item = test_negative_items[neg_mask]

                    user_tensor = torch.tensor(user, device=self.device)

                    test_recall_tmp, test_ndcg_tmp = user_model.evaluate(user_tensor.unsqueeze(0), test_item,
                                                                         user_tensor.unsqueeze(0),
                                                                         test_negative_item)
                    test_recall += test_recall_tmp
                    test_ndcg += test_ndcg_tmp
                if isinstance(user_model.model, NCF_client):
                    test_recall_large += test_recall_tmp
                    test_ndcg_large += test_ndcg_tmp
                    large += 1
                elif isinstance(user_model.model, NCF_client_low2):
                    test_recall_medium += test_recall_tmp
                    test_ndcg_medium += test_ndcg_tmp
                    medium += 1
                elif isinstance(user_model.model, NCF_client_low4):
                    test_recall_small += test_recall_tmp
                    test_ndcg_small += test_ndcg_tmp
                    small += 1
                else:
                    raise ValueError("Unknown client model type")
                # logging.info(f"The recall of {user} client is: {test_recall_tmp} ")

            test_recall /= len(participants)
            test_ndcg /= len(participants)

            if large != 0:
                test_recall_large /= large
                test_ndcg_large /= large

            if medium != 0:
                test_recall_medium /= medium
                test_ndcg_medium /= medium

            if small != 0:
                test_recall_small /= small
                test_ndcg_small /= small

        # Local Training
        local_train_times = []
        for user in participants:
            client = self.clients[user]
            # load current user's training data and instance a train loader.
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)

            # Measure the execution time of local_train.
            start_time_local_train = time.time()
            loss = client.local_train(user_dataloader)
            end_time_local_train = time.time()
            local_train_time = end_time_local_train - start_time_local_train
            local_train_times.append(local_train_time)

            all_loss += loss
            # user_param_changes[user] = param_changes

        avg_local_train_time = sum(local_train_times) / len(local_train_times)
        logging.info(f"local_train avg execte time: {avg_local_train_time} 秒")

        # Federate
        start_time_federate = time.time()
        self.federate(participants)
        end_time_federate = time.time()
        federate_time = end_time_federate - start_time_federate
        logging.info(f"federate execute time: {federate_time} 秒")
        return all_loss, test_recall, test_ndcg, test_recall_large, test_recall_medium, test_recall_small, test_ndcg_large, test_ndcg_medium, test_ndcg_small

    def evaluate(self, evaluate_data):
        test_recall = 0
        test_ndcg = 0

        test_users, test_items = evaluate_data[0], evaluate_data[1]
        test_negative_users, test_negative_items = evaluate_data[2], evaluate_data[3]

        if self.config['use_cuda']:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            test_negative_users = test_negative_users.cuda()
            test_negative_items = test_negative_items.cuda()

        for user in range(self.config['num_users']):
            user_model = self.clients[user]
            with torch.no_grad():
                # Obtain the positive samples of the current user
                # (the positive samples in the test_items that are relevant to the current user)
                user_mask = test_users == user
                test_item = test_items[user_mask]

                neg_mask = test_negative_users == user
                test_negative_item = test_negative_items[neg_mask]

                user_tensor = torch.tensor(user, device=self.device)

                test_recall_tmp, test_ndcg_tmp = user_model.evaluate(user_tensor.unsqueeze(0), test_item,
                                                                     user_tensor.unsqueeze(0), test_negative_item)
                test_recall += test_recall_tmp
                test_ndcg += test_ndcg_tmp

        test_recall /= len(self.clients)
        test_ndcg /= len(self.clients)
        return test_recall, test_ndcg

    # test selected clients
    def selected_indices_evaluate(self, evaluate_data, indices):
        test_recall = 0
        test_ndcg = 0

        test_users, test_items = evaluate_data[0], evaluate_data[1]
        test_negative_users, test_negative_items = evaluate_data[2], evaluate_data[3]

        if self.config['use_cuda']:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            test_negative_users = test_negative_users.cuda()
            test_negative_items = test_negative_items.cuda()

        for user in indices:
            user_model = self.clients[user]
            with torch.no_grad():
                user_mask = test_users == user
                test_item = test_items[user_mask]

                neg_mask = test_negative_users == user
                test_negative_item = test_negative_items[neg_mask]

                user_tensor = torch.tensor(user, device=self.device)

                test_recall_tmp, test_ndcg_tmp = user_model.evaluate(user_tensor.unsqueeze(0), test_item,
                                                                     user_tensor.unsqueeze(0), test_negative_item)
                test_recall += test_recall_tmp
                test_ndcg += test_ndcg_tmp

        test_recall /= len(indices)
        test_ndcg /= len(indices)
        return test_recall, test_ndcg

    def federate(self, participants):
        server_state_dict = self.server_model.state_dict()
        old_server_state_dict = server_state_dict
        for key in server_state_dict.keys():
            if 'autoencoder' in key:
                continue  # Skip autoencoder parameters
            client_params = []
            for user in participants:
                client = self.clients[user]
                if isinstance(client.model, NCF_client):
                    client_params.append(client.model.state_dict()[key])
                elif isinstance(client.model, NCF_client_low2):
                    if key == 'embedding_item.weight':
                        client_params.append(
                            self.server_model.autoencoder_item_low2.decoder(client.model.state_dict()[key]))
                    elif key == 'mlp.0.weight':
                        client_params.append(
                            self.server_model.autoencoder_mlp_first_low2.decoder(client.model.state_dict()[key]))
                    else:
                        client_params.append(client.model.state_dict()[key])
                elif isinstance(client.model, NCF_client_low4):
                    if key == 'embedding_item.weight':
                        client_params.append(
                            self.server_model.autoencoder_item_low4.decoder(client.model.state_dict()[key]))
                    elif key == 'mlp.0.weight':
                        client_params.append(
                            self.server_model.autoencoder_mlp_first_low4.decoder(client.model.state_dict()[key]))
                    else:
                        client_params.append(client.model.state_dict()[key])
                else:
                    raise ValueError("Unknown client model type")

            server_state_dict[key] = torch.stack(client_params, 0).mean(0)
        self.server_model.load_state_dict(server_state_dict)

    def client_change(self, pre_mapping, post_mapping, all_train_data, evaluate_data, round_id):
        logging.info('-' * 80)
        logging.info('-' * 80)
        pre_clients, client_indices = self.select_clients_by_type(pre_mapping)
        # print(client_indices)
        pre_recall, pre_ndcg = self.selected_indices_evaluate(evaluate_data, client_indices)
        logging.info(
            '[Epoch {}: Before change, SELECTED clients model from {} to {}] These {} Clients Recall = {:.4f}, NDCG = {:.4f}'.format(
                round_id, pre_mapping,
                post_mapping, pre_mapping,
                pre_recall, pre_ndcg))
        post_type = ''
        if post_mapping == 'small':
            post_type = 'NCF_client_low4'
        elif post_mapping == 'medium':
            post_type = 'NCF_client_low2'
        elif post_mapping == 'large':
            post_type = 'NCF_client'
        post_clients = []
        server = copy.deepcopy(self.server_model)
        for user in client_indices:
            # print(user)
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            user_model = self.clients[user]
            loss = user_model.change_model(post_type, user_dataloader, self.config['temperature'], self.config['alpha'],
                                           server)
            # logging.info(
            #     '[Epoch {}: Change Clients model {} from {} to {}] Loss = {:.4f} '.format(
            #         round_id, user,
            #         pre_mapping,
            #         post_mapping,
            #         loss))
        post_recall, post_ndcg = self.selected_indices_evaluate(evaluate_data, client_indices)
        logging.info(
            '[Epoch {}: After change, SELECTED clients model from {} to {}] These {} Clients Recall = {:.4f}, NDCG = {:.4f}'.format(
                round_id, pre_mapping,
                post_mapping, post_mapping,
                post_recall, post_ndcg))

    def random_client_change(self, all_train_data, evaluate_data, round_id):
        logging.info('-' * 80)
        logging.info('-' * 80)

        client_indices = random.sample(range(self.config['num_users']), int(
            self.config['continued_user_migration_user_ratio'] * self.config['num_users']))

        # get the model type of the current client
        pre_hit_ratio, pre_ndcg = self.selected_indices_evaluate(evaluate_data, client_indices)
        logging.info(
            '[Epoch {}: Before Random change, SELECTED clients model] HR = {:.4f}, NDCG = {:.4f}'.format(
                round_id,
                pre_hit_ratio, pre_ndcg))

        # Define the type of the target model
        post_type_mapping = {
            'small': 'NCF_client_low4',
            'medium': 'NCF_client_low2',
            'large': 'NCF_client'
        }

        # Randomly select the target model type, ensuring that it is not the current model type.
        server = copy.deepcopy(self.server_model)

        for user in client_indices:
            user_model = self.clients[user]

            current_model_type = user_model.selected_model_type

            # Randomly select the target model type, ensuring that it is not the current model type.
            available_post_types = [t for t in post_type_mapping.values() if t != current_model_type]
            post_type = random.choice(available_post_types)

            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)

            loss = user_model.change_model(post_type, user_dataloader, self.config['temperature'], self.config['alpha'],
                                           server)
            # logging.info(
            #     '[Epoch {}: Change Clients model {} from {} to {}] Loss = {:.4f} '.format(
            #         round_id, user,
            #         pre_mapping,
            #         post_mapping,
            #         loss))

        post_hit_ratio, post_ndcg = self.selected_indices_evaluate(evaluate_data, client_indices)
        logging.info(
            '[Epoch {}: After change, SELECTED clients model ] HR = {:.4f}, NDCG = {:.4f}'.format(
                round_id,
                post_hit_ratio, post_ndcg))

    def select_clients_by_type(self, type):
        if type == 'small':
            selected_clients = [client for client in self.clients if isinstance(client.model, NCF_client_low4)]
        elif type == 'medium':
            selected_clients = [client for client in self.clients if isinstance(client.model, NCF_client_low2)]
        elif type == 'large':
            selected_clients = [client for client in self.clients if isinstance(client.model, NCF_client)]
        else:
            raise ValueError(f"Unknown mapping: {type}")

        client_indices = [idx for idx, client in enumerate(self.clients) if client in selected_clients]
        return selected_clients, client_indices

    def save(self, alias, epoch_id, recall, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, recall, ndcg)
        save_checkpoint(self.server_model, model_dir)

    def add_noise(self, params, noise_multiplier, max_grad_norm):
        # Generate Gaussian noise
        noise = torch.normal(0, noise_multiplier * max_grad_norm, size=params.shape, device=params.device)
        print("params:{}", params)
        print("noise:{}", noise)
        print("noise:{}", noise)
        return params + noise
