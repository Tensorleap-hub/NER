from typing import Tuple, Optional, List, Union

import numpy
from datasets import load_dataset
import numpy as np
from datasets import Dataset


from NER.config import CONFIG

np.random.seed(0)

def load_data() -> Tuple[Dataset, Dataset, Dataset]:


    ds_train = load_dataset("eriktks/conll2003", cache_dir=CONFIG['cache_dir'], split="train")
    ds_val = load_dataset("eriktks/conll2003", cache_dir=CONFIG['cache_dir'], split="validation")
    ds_test = load_dataset("eriktks/conll2003", cache_dir=CONFIG['cache_dir'], split="test")
    return ds_train, ds_val, ds_test


def downsample_hf_dataset(dataset: Dataset, size: int, indices: Optional[Union[List[int], np.ndarray]] = None) -> Dataset:

    indices = np.array(indices) if indices is not None else np.arange(len(dataset))

    # Generate random indices without replacement
    random_indices = np.random.choice(indices, size=min(size, len(dataset)), replace=False)

    # Subsample the dataset using the selected indices
    subset_dataset = dataset.select(random_indices)

    return subset_dataset



