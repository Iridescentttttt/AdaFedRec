import random
import sys

import pandas as pd
import numpy as np
import datetime
import os
from NCF import NCFEngine
from data import SampleGenerator
from utils import *
import pickle

random.seed(12345)
# Training settings
config = {
    # Dateset settings
    'dataset': 'ml-100k',
    'num_users': None,
    'num_items': None,
    'clients_sample_ratio': 1,
    'clients_sample_num': 256,
    'num_round': 1000,
    'local_epoch': 1,
    'test_per_epoch': 1,
    'log_path': 'log/ml-100k/',
    # Parameter settings
    'lr': 0.01,
    'lr_eta': 80,
    'batch_size': 256,
    'latent_dim': 32,
    'num_negative': 4,
    'l2_regularization': 0.0,
    'autoencoder_train_epoch': 75,
    'autoencoder_lr': 0.01,
    'random_client': True,
    'model_type_ratios':[1,1,1],
    # Experiment for User device migration
    'fixed_user_device_migration': False,
    'fixed_user_migration_epoch': 100,
    'continued_user_device_migration': False,
    'continued_user_migration_per_epoch': 20,
    'continued_user_migration_user_ratio':0.9,
    'temperature': 3.0,
    'alpha': 0.5,
    # 'beta': 0.33,
    'KD_epoch': 1,
    'original_type': 'large',
    'target_type': 'medium',
    # Device settings
    'use_cuda': True,
    'device_id': 0,
    'top_k':20
}

if config['dataset'] == 'ml-1m':
    config['num_users'] = 6040
    config['num_items'] = 3706
elif config['dataset'] == 'ml-100k':
    config['num_users'] = 943
    config['num_items'] = 1682
elif config['dataset'] == 'lastfm-2k':
    config['num_users'] = 1482
    config['num_items'] = 12399
elif config['dataset'] == 'amazon':
    config['num_users'] = 8072
    config['num_items'] = 11830
else:
    pass

engine = NCFEngine(config)

# Logging.
path = config['log_path']
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logname = os.path.join(path, current_time + '.txt')
initLogging(logname)
rating = None
# Load Data
dataset_dir = "./data/" + config['dataset'] + "/" + "ratings.dat"
if config['dataset'] == "ml-1m":
    rating = pd.read_csv(dataset_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                         engine='python')
elif config['dataset'] == "ml-100k":
    rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                         engine='python')
elif config['dataset'] == "lastfm-2k":
    rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                         engine='python')
elif config['dataset'] == "amazon":
    rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                         engine='python')
    rating = rating.sort_values(by='uid', ascending=True)
else:
    pass

# Reindex
user_id = rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
rating = pd.merge(rating, user_id, on=['uid'], how='left')
item_id = rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
rating = pd.merge(rating, item_id, on=['mid'], how='left')
rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
logging.info('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
logging.info('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))

# DataLoader for training
sample_generator = SampleGenerator(ratings=rating)

# all_train_data = sample_generator.store_all_train_data(config['num_negative'])
# test_data = sample_generator.test_data

train_loss_list = []

val_hr_list = []
val_ndcg_list = []
best_val_hr = 0
best_val_ndcg = 0
best_val_round = 0

test_recall_list = []
test_ndcg_list = []
best_test_recall = 0
best_test_ndcg = 0
best_test_round = 0

logging.info('FedMLP')
# Log the configuration settings
config_str = ', '.join([f'{key}: {value}' for key, value in config.items()])
logging.info(config_str)


################################################################################ save or load data
# Convert Tensor to numpy array
# test_data_np = [tensor.numpy() for tensor in test_data]
#
# # save data file
# with open('./data/'+config['dataset']+'/all_train_data.pkl', 'wb') as f:
#     pickle.dump(all_train_data, f)
#
#
# with open('./data/'+config['dataset']+'/test_data.pkl', 'wb') as f:
#     pickle.dump(test_data_np, f)
#
# sys.exit()

# load data file
with open('./data/'+config['dataset']+'/all_train_data.pkl', 'rb') as f:
    all_train_data = pickle.load(f)

with open('./data/'+config['dataset']+'/test_data.pkl', 'rb') as f:
    loaded_test_data_np = pickle.load(f)

# Convert the numpy array back to Tensor
test_data = [torch.from_numpy(arr) for arr in loaded_test_data_np]
################################################################################ save or load data
_,small = engine.select_clients_by_type("small")
_,medium = engine.select_clients_by_type("medium")
_,large = engine.select_clients_by_type("large")
logging.info('Small Clients :{}'.format(small))
logging.info('Medium Clients :{}'.format(medium))
logging.info('Large Clients :{}'.format(large))

for round in range(config['num_round']):
    logging.info('-' * 80)
    logging.info('Round {} starts !'.format(round))
    # all_train_data = sample_generator.store_all_train_data(config['num_negative'])

    # logging.info('Training phase!')
    tr_loss,recall,ndcg,test_recall_large, test_recall_medium, test_recall_small, test_ndcg_large, test_ndcg_medium, test_ndcg_small = engine.fed_train_a_round(all_train_data,test_data,round_id=round)
    train_loss_list.append(tr_loss)
    logging.info('[Training Epoch {}] tr_loss = {:.4f}'.format(round, tr_loss))

    # logging.info('[Epoch {}] Test_Recall = {:.4f}, Test_NDCG = {:.4f}'.format(round, recall,ndcg))
    logging.info(
        '[Epoch {}] Test_Recall = {:.4f}, Test_NDCG = {:.4f},\n'
        'test_recall_large = {:.4f}, test_recall_medium = {:.4f}, test_recall_small = {:.4f}, test_ndcg_large = {:.4f}, test_ndcg_medium = {:.4f}, test_ndcg_small = {:.4f}'.format(
            round,
            recall, ndcg,
            test_recall_large, test_recall_medium, test_recall_small, test_ndcg_large, test_ndcg_medium, test_ndcg_small))

    if recall >= best_test_recall:
        best_test_recall = recall
        best_test_ndcg = ndcg
        best_test_round = round
    logging.info(
        'The Best Test Metrics is at [Epoch {}] Recall = {:.4f}, NDCG = {:.4f}'.format(best_test_round, best_test_recall,
                                                                                   best_test_ndcg))

    if config['fixed_user_device_migration'] == True and round == config['fixed_user_migration_epoch']:
        engine.client_change(config['original_type'], config['target_type'], all_train_data, test_data, round)
        recall, ndcg = engine.evaluate(test_data)
        logging.info('[After fixed clients migration, Testing Epoch {}] All clients: Recall = {:.4f}, NDCG = {:.4f}'.format(round,
                                                                                                                  recall,
                                                                                                                  ndcg))
        logging.info('-' * 80)

    if config['continued_user_device_migration'] == True and round % config['continued_user_migration_per_epoch'] == 0:
        engine.random_client_change( all_train_data, test_data, round)
        recall, ndcg = engine.evaluate(test_data)
        logging.info('[After random clients migration, Testing Epoch {}] All clients: Recall = {:.4f}, NDCG = {:.4f}'.format(round,
                                                                                                                  recall,
                                                                                                                  ndcg))
        logging.info('-' * 80)

logging.info('recall_list: {}'.format(test_recall_list))
logging.info('ndcg_list: {}'.format(test_ndcg_list))
logging.info('Best test recall: {}, ndcg: {} at round {}'.format(best_test_recall,
                                                             best_test_ndcg,
                                                             best_test_round))