from .statistical import TopPersonalRecommender, TopPopularRecommender
from .up_cf import UPCFRecommender
from .tifuknn import (
    TIFUKNNRecommender,
    TIFUKNNTimeDaysRecommender,
    TIFUKNNTimeDaysNextTsRecommender,
    TIFUKNNRecommenderWaves
)
from .core import IRecommender, IRecommenderNextTs


MODELS = {
    "top_popular": TopPopularRecommender,
    "top_personal": TopPersonalRecommender,
    "up_cf": UPCFRecommender,
    "tifuknn": TIFUKNNRecommender,
    "tifuknn_time_days": TIFUKNNTimeDaysRecommender,
    "tifuknn_time_days_next_ts": TIFUKNNTimeDaysNextTsRecommender,
    "tifuknn_modified_weights": TIFUKNNRecommenderWaves,
    "tifuknn_time_days_modified_weights": TIFUKNNRecommenderWaves
}
