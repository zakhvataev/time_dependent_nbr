from .base import IMetric
from .recall import Recall
from .ndcg import NDCG
from .diversity import Diversity_MeanInterList


METRICS = {
    Recall.metric_name: Recall,
    NDCG.metric_name: NDCG,
    Diversity_MeanInterList.metric_name: Diversity_MeanInterList 
}
