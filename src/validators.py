import math

from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import BaseCrossValidator
from dateutil.relativedelta import relativedelta
from config import RANDOM_SEED

class TimeGroupedKFold(BaseCrossValidator):
    def __init__(self, n_splits=3, val_data_share=0.3):

        self.n_splits = n_splits
        self.val_data_share = val_data_share

    def split(self, X, y=None, groups=None):
        all_time_coh = sorted(groups.unique())
        n_time_coh = len(all_time_coh)

        quant_size = math.ceil(n_time_coh * self.val_data_share / self.n_splits)
        train_size = quant_size * math.floor(
            n_time_coh * (1 - self.val_data_share) / quant_size
        )

        i = 0
        for i in range(self.n_splits - 1):
            train_index = groups[
                groups.isin(
                    all_time_coh[i * quant_size : (i * quant_size + train_size)]
                )
            ].index.values
            test_index = groups[
                groups.isin(
                    all_time_coh[
                        (i * quant_size + train_size) : (
                            (i + 1) * quant_size + train_size
                        )
                    ]
                )
            ].index.values
            yield train_index, test_index

        train_index = groups[
            groups.isin(all_time_coh[(i + 1) * quant_size : -quant_size])
        ].index.values
        test_index = groups[groups.isin(all_time_coh[-quant_size:])].index.values
        yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
class SimpleSplitter(BaseCrossValidator):
    
    def __init__(self, n_splits=3, val_unique_groups=3, train_accounts_share=0, gap_unique_groups=0):
        self.gap = gap_unique_groups
        self.quant = val_unique_groups
        self.train_accounts_share = train_accounts_share
        self.n_splits = n_splits

    def split(self, X, y, groups):
        sorted_unique_gropus = sorted(groups.unique())
        train_size = len(sorted_unique_gropus) - self.n_splits*self.quant - self.gap
        train_start_index = 0
        for i in range(self.n_splits):
            train_end_index = train_start_index + train_size - 1
            val_start_index = train_end_index + self.gap
            val_end_index = val_start_index + self.quant
            train_mask = (groups >= sorted_unique_gropus[train_start_index]) \
                          & (groups <= sorted_unique_gropus[train_end_index])
            X_masked = X[train_mask]
            if self.train_accounts_share > 0:
                X_masked_sel, _, = train_test_split(X_masked,
                                                    stratify=y[train_mask],
                                                    test_size=self.train_accounts_share,
                                                    random_state=RANDOM_SEED)
                training_ids = set(X_masked_sel.account_id.unique())
                train_mask = train_mask & (X['account_id'].isin(training_ids))
            test_mask = (groups > sorted_unique_gropus[val_start_index]) \
                        & (groups <= sorted_unique_gropus[val_end_index])
            if self.train_accounts_share > 0:
                test_mask = test_mask & (~X['account_id'].isin(training_ids))
            train_index = groups[train_mask].index.values
            test_index = groups[test_mask].index.values
            train_start_index += self.quant
            yield train_index, test_index
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
