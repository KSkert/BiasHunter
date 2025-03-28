from LN import FairnessTester  # assuming this code is saved as fairness_tester.py

hparams = {
    'file_path': "dataset/processed_credit.csv",
    'model_path': "DNN/model_processed_credit.h5",
    'sensitive': ['SEX', 'AGE'], 
    'target': 'class',
    'max_combo_pairs': 20,          # defines the maximum number of group pairings 
                                    # (e.g., ('Female', 'White') vs ('Male', 'Black')) 
                                    # that will be tested for each sensitive feature combination.

    'combination_length': 2,        # controls the number of sensitive features considered *together*

    'pairs_per_combo_pair_pctg': 10 # determines the sampling rate for synthetic test pairs within 
                                    # each group pairing, expressed as a percentage of the total dataset size.
}

tester = FairnessTester(hparams)
results = tester.run_tests()
for res in results:
    print(f"IDI Ratio: {res['IDI_ratio']}")
    print("-" * 40)
    print("Sensitive features used for discrimination testing:", res['sensitive_features'])
    print("-" * 40)
    print(f"Generated samples represent {res['percent_generated']:.2f}% of the whole dataset")
    print()
