import numpy as np
import pandas as pd


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Calculate a nDCG score for a given user."""
    y_max_sorted = y_true[y_true.argsort()[::-1]]
    y_true_sorted = y_true[y_score.argsort()[::-1]]

    num_items = y_true.shape[0]
    k = num_items if num_items < k else k

    dcg_score = 2 ** (y_true_sorted[0] - 1)
    for i in np.arange(1, k):
        dcg_score += (2 ** (y_true_sorted[i] - 1)) / np.log2(i + 2)

    max_score = 2 ** (y_max_sorted[0] - 1)
    for i in np.arange(1, k):
        max_score += (2 ** (y_max_sorted[i] - 1)) / np.log2(i + 2)

    return dcg_score / max_score


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Calculate a Recall score for a given user."""
    y_true_sorted = y_true[y_score.argsort()[::-1]]

    num_items = y_true.shape[0]
    k = num_items if num_items < k else k

    return np.sum(y_true_sorted[:k]) / np.sum(y_true)


class PredictRankings:
    """Predict rankings by trained recommendations."""

    def __init__(
        self, user_embed: np.ndarray, item_embed: np.ndarray, item_bias: np.ndarray
    ) -> None:
        """Initialize Class."""
        self.user_embed = user_embed
        self.item_embed = item_embed
        self.item_bias = item_bias

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        scores = (user_emb @ item_emb.T).flatten() + self.item_bias[items]
        return scores


metrics = {
    "nDCG": ndcg_at_k,
    "Recall": recall_at_k,
}


def aoa_evaluator(
    user_embed: np.ndarray,
    item_embed: np.ndarray,
    item_bias: np.ndarray,
    test: np.ndarray,
    model_name: str,
    at_k: int = 5,
) -> pd.DataFrame:
    """Calculate ranking metrics with average-over-all evaluator."""
    # test data
    users = test[:, 0]
    items = test[:, 1]
    ratings = test[:, 2]

    # define model
    model = PredictRankings(
        user_embed=user_embed, item_embed=item_embed, item_bias=item_bias
    )

    results = {}
    for metric in metrics:
        results[metric] = []
    # calculate ranking metrics
    np.random.seed(12345)
    for user in set(users):
        indices = users == user
        items_for_user = items[indices]
        ratings_for_user = ratings[indices]

        # predict ranking score for each user
        scores = model.predict(users=user, items=items_for_user)
        for metric, metric_func in metrics.items():
            results[metric].append(metric_func(ratings_for_user, scores, at_k))

    # aggregate results
    results_df = pd.DataFrame(index=results.keys())
    results_df[model_name] = list(map(np.mean, list(results.values())))

    return results_df
