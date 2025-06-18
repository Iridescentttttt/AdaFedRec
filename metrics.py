import numpy as np
import pandas as pd


class MetronAtK:
    def __init__(self, top_k):
        """
        Evaluation class for Recall@K and NDCG@K

        Args:
            top_k (int): Number of top recommended items to evaluate.
        """
        self._top_k = top_k
        self.test_items_dict = None  # Dictionary to store ground truth items for each user
        self._subjects = None  # Evaluation data: combined prediction results for test and negative samples

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        Set the evaluation dataset

        Args:
            subjects (list): A list of 6 elements:
                [test_users, test_items, test_scores, neg_users, neg_items, neg_scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]

        # Construct a dictionary of ground truth items per user
        self.test_items_dict = {}
        for user, item in zip(test_users, test_items):
            if user not in self.test_items_dict:
                self.test_items_dict[user] = set()
            self.test_items_dict[user].add(item)

        # Merge positive and negative samples into one DataFrame
        full = pd.DataFrame({
            'user': neg_users + test_users,
            'item': neg_items + test_items,
            'score': neg_scores + test_scores
        })

        # Rank items per user based on predicted scores (higher is better)
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)

        # Sort by user and rank for consistent evaluation
        self._subjects = full.sort_values(['user', 'rank'], ascending=[True, True])

    def cal_recall(self):
        """
        Calculate Recall@K

        Returns:
            float: Average Recall across all users.
        """
        full = self._subjects
        top_k = self._top_k

        # Get top-K items per user
        top_k_predictions = full[full['rank'] <= top_k]

        hits = 0  # Number of correctly recommended (true positive) items
        total_pos_items = 0  # Total number of ground truth items across all users

        for user, group in top_k_predictions.groupby('user'):
            predicted_items = set(group['item'].tolist())
            true_items = self.test_items_dict.get(user, set())

            hits += len(predicted_items & true_items)
            total_pos_items += len(true_items)

        recall = hits / total_pos_items if total_pos_items > 0 else 0.0
        return recall

    def cal_ndcg(self):
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)

        Returns:
            float: Average NDCG across all users.
        """
        full = self._subjects
        top_k = self._top_k
        top_k_predictions = full[full['rank'] <= top_k]

        ndcg_sum = 0.0
        total_users = 0

        for user, group in top_k_predictions.groupby('user'):
            predicted_items = group['item'].tolist()
            true_items = self.test_items_dict.get(user, set())

            dcg = 0.0  # Discounted Cumulative Gain
            idcg = 0.0  # Ideal DCG

            for i, item in enumerate(predicted_items[:top_k]):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)

            for i in range(min(len(true_items), top_k)):
                idcg += 1.0 / np.log2(i + 2)

            ndcg_sum += dcg / idcg if idcg > 0 else 0.0
            total_users += 1

        return ndcg_sum / total_users if total_users > 0 else 0.0
