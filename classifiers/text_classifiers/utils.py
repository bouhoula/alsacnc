import os
import random
from typing import List, Optional

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def get_loss_weights(
    labels: List, problem_type: str, use_loss_weights: bool
) -> Optional[List[float]]:
    if not use_loss_weights:
        return None
    if problem_type == "single_label_classification":
        return list(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(labels), y=labels
            )
        )
    else:
        assert problem_type == "multi_label_classification"
        counts = np.sum(labels, axis=0)
        print(counts)
        num_samples = counts.sum()
        return list(map(lambda x: (num_samples - x) / x, counts))
