import sys

import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
random.seed(0)

class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples for NCF learning
        # 99 negatives for each user's test item
        self.negatives = self._sample_negative(ratings)
        # divide all ratings into train and test two dataframes, which consit of userId, itemId and rating three columns.
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """
        Split the dataset into 80% training and 20% testing for each user.

        This function performs a per-user split to ensure that every user is
        represented in both the training and test sets. This is useful for
        personalized recommendation tasks to preserve user profiles.

        Args:
            ratings (pd.DataFrame): Input ratings DataFrame with at least
                                    columns ['userId', 'itemId', 'rating'].

        Returns:
            tuple: (train_data, test_data)
                train_data (DataFrame): 80% of user interactions.
                test_data (DataFrame): 20% of user interactions.
        """
        ratings = deepcopy(ratings)  # Prevent modifications to the original data

        train_list = []
        test_list = []

        # Group ratings by user
        grouped_ratings = ratings.groupby('userId')

        # For each user, perform an 80/20 train-test split
        for _, group in grouped_ratings:
            train, test = train_test_split(group, test_size=0.2, random_state=42)
            train_list.append(train)
            test_list.append(test)

        # Concatenate individual user splits into full training and testing sets
        train_data = pd.concat(train_list)
        test_data = pd.concat(test_list)

        # Ensure that all users appear in both sets (no missing users)
        assert train_data['userId'].nunique() == test_data['userId'].nunique()

        # Return only the necessary columns
        return train_data[['userId', 'itemId', 'rating']], test_data[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 198))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def store_all_train_data(self, num_negatives):
        """store all the train data as a list including users, items and ratings. each list consists of all users'
        information, where each sub-list stores a user's positives and negatives"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x,
                                                                                                   num_negatives))  # include userId, itemId, rating, negative_items and negatives five columns.
        single_user = []
        user_item = []
        user_rating = []
        # split train_ratings into groups according to userId.
        grouped_train_ratings = train_ratings.groupby('userId')
        train_users = []
        for userId, user_train_ratings in grouped_train_ratings:
            train_users.append(userId)
            user_length = len(user_train_ratings)
            for row in user_train_ratings.itertuples():
                single_user.append(int(row.userId))
                user_item.append(int(row.itemId))
                user_rating.append(float(row.rating))
                for i in range(num_negatives):
                    single_user.append(int(row.userId))
                    user_item.append(int(row.negatives[i]))
                    user_rating.append(float(0))  # negative samples get 0 rating
            assert len(single_user) == len(user_item) == len(user_rating)
            assert (1 + num_negatives) * user_length == len(single_user)
            users.append(single_user)
            items.append(user_item)
            ratings.append(user_rating)
            single_user = []
            user_item = []
            user_rating = []
        assert len(users) == len(items) == len(ratings) == len(self.user_pool)
        assert train_users == sorted(train_users)
        # print(train_users)
        # print(sorted(train_users))
        # print(len(train_users))

        return [users, items, ratings]

    @property
    def validate_data(self):
        """create validation data"""
        val_ratings = pd.merge(self.val_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        val_users, val_items, negative_users, negative_items = [], [], [], []
        for row in val_ratings.itertuples():
            val_users.append(int(row.userId))
            val_items.append(int(row.itemId))
            for i in range(int(len(row.negative_samples) / 2)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        assert len(val_users) == len(val_items)
        assert len(negative_users) == len(negative_items)
        assert 99 * len(val_users) == len(negative_users)
        assert val_users == sorted(val_users)
        return [torch.LongTensor(val_users), torch.LongTensor(val_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]

    @property
    def test_data(self):
        """
        Create evaluation data with negative samples.

        For each user, negative items are all items except those that appear
        in the user's training and testing interactions. These negative samples
        are used to evaluate top-K recommendation performance (e.g., Recall@K, NDCG@K).

        Returns:
            list: Four torch.LongTensor objects:
                [test_users, test_items, negative_users, negative_items]
        """
        # Set of all item IDs in the dataset
        all_items = set(self.item_pool)

        # Merge test ratings with precomputed user-specific negative sample lists
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        train_ratings = self.train_ratings

        test_users, test_items = [], []
        negative_users, negative_items = [], []

        # Build a dictionary: user -> set of positive items in training data
        user_train_positive_items = train_ratings.groupby('userId')['itemId'].apply(set).to_dict()

        # Build a dictionary: user -> set of positive items in test data
        user_test_positive_items = test_ratings.groupby('userId')['itemId'].apply(set).to_dict()

        # Create a dictionary of user -> set of negative items
        # Negative items = all items - (user's train items + test items)
        user_negative_items = {}
        for user, test_items_set in user_test_positive_items.items():
            train_items_set = user_train_positive_items.get(user, set())
            neg_items = all_items - test_items_set - train_items_set
            user_negative_items[user] = neg_items

        # Loop over test_ratings and build user-item positive and negative lists
        for row in test_ratings.itertuples():
            user = int(row.userId)
            positive_item = int(row.itemId)

            # Append positive sample
            test_users.append(user)
            test_items.append(positive_item)

            # Append corresponding negative items (once per user)
            if user in user_negative_items:
                neg_items = user_negative_items[user]
                negative_users.extend([user] * len(neg_items))
                negative_items.extend(list(neg_items))

                # Avoid duplicate addition of negative items
                del user_negative_items[user]

        # Sanity check
        assert len(test_users) == len(test_items)
        assert len(negative_users) == len(negative_items)

        return [
            torch.LongTensor(test_users),
            torch.LongTensor(test_items),
            torch.LongTensor(negative_users),
            torch.LongTensor(negative_items)
        ]
