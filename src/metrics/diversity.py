import numpy as np
from .base import IMetric


class Diversity_MeanInterList(IMetric):
    metric_name: str = "diversity"

    def __init__(self, topk, n_items):
        super().__init__(topk=topk)

        self.recommended_counter = np.zeros(n_items, dtype=np.float64)

        self.n_evaluated_users = 0
        self.n_items = n_items
        self.cumulative_value = 0.0
        self.cutoff = topk

    def add_recommendations(self, model_scores):
        self.model_scores = model_scores.copy()
        self.recommended_items_ids = model_scores.argsort()[-self.cutoff:]

        assert len(self.recommended_items_ids) <= self.cutoff, "Diversity_MeanInterList: recommended list contains less elements than cutoff"

        self.n_evaluated_users += 1

        if len(self.recommended_items_ids) > 0:
            self.recommended_counter[self.recommended_items_ids-1] += 1

    def get_metric_value(self):

        # Requires to compute the number of common elements for all couples of users
        if self.n_evaluated_users == 0:
            return 1.0

        cooccurrences_cumulative = np.sum(self.recommended_counter**2) - self.n_evaluated_users*self.cutoff

        # All user combinations except diagonal
        all_user_couples_count = self.n_evaluated_users**2 - self.n_evaluated_users

        diversity_cumulative = all_user_couples_count - cooccurrences_cumulative/self.cutoff

        self.cumulative_value = diversity_cumulative/all_user_couples_count

        return self.cumulative_value

    def merge_with_other(self, other_metric_object):

        assert other_metric_object is Diversity_MeanInterList, "Diversity_MeanInterList: attempting to merge with a metric object of different type"

        assert np.all(self.recommended_counter >= 0.0), "Diversity_MeanInterList: self.recommended_counter contains negative counts"
        assert np.all(other_metric_object.recommended_counter >= 0.0), "Diversity_MeanInterList: other.recommended_counter contains negative counts"

        self.recommended_counter += other_metric_object.recommended_counter
        self.n_evaluated_users += other_metric_object.n_evaluated_users

    def reset(self):
        self.cumulative_value = 0.0
        self.n_users = 0