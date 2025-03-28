import os
from itertools import combinations
import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import load_model

class FairnessTester:
    def __init__(self, hparams):
        self.file_path = hparams['file_path']
        self.model_path = hparams['model_path']
        self.sensitive = hparams['sensitive']
        self.num_sensitive_features = hparams['combination_length']
        self.target_name = hparams['target']
        self.max_combo_pairs = hparams.get('max_combo_pairs', 10)
        self.pairs_per_combo_pair_pctg = hparams.get('pairs_per_combo_pair_pctg', 2.0)

        self._load_data()
        self._load_model()

        self.dataset_size = len(self.X)
        self._build_sorted_value_lists()

        self.synthetic_instances = []

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        if self.target_name in df.columns:
            self.y = df[self.target_name].values
            self.X = df.drop(columns=[self.target_name])
        else:
            raise ValueError(f"Target column '{self.target_name}' not found in dataset.")

    def _load_model(self):
        self.model = load_model(self.model_path)

    def _model_predict(self, inputs):
        raw_preds = self.model.predict(inputs)
        return (raw_preds > 0.5).astype(int).flatten()

    def _build_group_dict(self, sens_cols):
        group_dict = {}
        data_matrix = self.X.values
        for i, row in enumerate(data_matrix):
            key = tuple(row[self.X.columns.get_loc(c)] for c in sens_cols)
            group_dict.setdefault(key, []).append(i)
        return group_dict

    def _build_sorted_value_lists(self):
        self.sorted_values_per_col = {}
        for col in self.X.columns:
            self.sorted_values_per_col[col] = sorted(self.X[col].unique())

    def _perturb_to_nearest(self, value, sorted_list):
        if value not in sorted_list:
            return value
        idx = sorted_list.index(value)
        options = []
        if idx > 0:
            options.append(sorted_list[idx - 1])
        options.append(value)
        if idx < len(sorted_list) - 1:
            options.append(sorted_list[idx + 1])
        return random.choice(options)

    def _generate_pairs_for_combo_pair(self, groupA_indices, groupB_indices, pairs_needed, non_sens_cols_indices):
        data_matrix = self.X.values
        disc_count = 0
        pairs_tested = 0
        attempts = 0
        max_attempts = pairs_needed * 10

        while pairs_tested < pairs_needed and attempts < max_attempts:
            attempts += 1
            if not groupA_indices or not groupB_indices:
                break
            idxA = random.choice(groupA_indices)
            idxB = random.choice(groupB_indices)

            xA = data_matrix[idxA].copy()
            xB = data_matrix[idxB].copy()

            for _ in range(3):
                if not non_sens_cols_indices:
                    break
                column_choice = random.choice(non_sens_cols_indices)
                col_name = self.X.columns[column_choice]
                sorted_vals = self.sorted_values_per_col[col_name]
                xA[column_choice] = self._perturb_to_nearest(xA[column_choice], sorted_vals)
                xB[column_choice] = self._perturb_to_nearest(xB[column_choice], sorted_vals)

            preds = self._model_predict(np.array([xA, xB]))
            if preds[0] != preds[1]:
                disc_count += 1

            # Save synthetic instances
            self.synthetic_instances.append(xA.tolist())
            self.synthetic_instances.append(xB.tolist())

            pairs_tested += 1

        return pairs_tested, disc_count


    def _test_combo_pairs(self, sens_cols):
        group_dict = self._build_group_dict(sens_cols)
        combos = list(group_dict.keys())
        if len(combos) < 2:
            return 0, 0

        all_combo_pairs = [(combos[i], combos[j]) for i in range(len(combos)) for j in range(i + 1, len(combos))]
        random.shuffle(all_combo_pairs)
        combo_pairs_to_use = all_combo_pairs[:self.max_combo_pairs]

        pairs_needed = int(np.ceil((self.pairs_per_combo_pair_pctg / 100.0) * self.dataset_size))

        non_sens_cols = [c for c in self.X.columns if c not in sens_cols]
        non_sens_cols_indices = [self.X.columns.get_loc(c) for c in non_sens_cols]

        total_pairs_tested = 0
        disc_count = 0

        for (kA, kB) in combo_pairs_to_use:
            groupA_indices = group_dict[kA]
            groupB_indices = group_dict[kB]
            p_tested, p_disc = self._generate_pairs_for_combo_pair(groupA_indices, groupB_indices, pairs_needed, non_sens_cols_indices)
            total_pairs_tested += p_tested
            disc_count += p_disc

        return total_pairs_tested, disc_count

    def run_tests(self):
        results = []
        for sens_comb in combinations(self.sensitive, self.num_sensitive_features):
            total_pairs, disc_pairs = self._test_combo_pairs(sens_comb)

            if total_pairs == 0:
                idi_ratio = 0.0
                percent_gen = 0.0
            else:
                total_inputs = 2 * total_pairs
                idi_count = 2 * disc_pairs
                idi_ratio = idi_count / float(total_inputs)
                percent_gen = (total_inputs / self.dataset_size) * 100.0

            results.append({
                "IDI_ratio": idi_ratio,
                "sensitive_features": sens_comb,
                "percent_generated": percent_gen
            })

        if self.synthetic_instances:
            synthetic_df = pd.DataFrame(self.synthetic_instances, columns=self.X.columns)
            base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
            synthetic_df.to_csv(f"{base_filename}_synthetic_instances.csv", index=False)

        return results
