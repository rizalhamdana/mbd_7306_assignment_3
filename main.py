from collaborative_filtering.CollaborativeFiltering import CollaborativeFiltering
from pattern_mining.PatternMining import FlexiblePatternMiner

from os import path

import pandas as pd

def main():
    DATA_PATH = 'data'
    TRAIN_FILE = 'Groceries data train.csv'
    TRAIN_PATH = path.join(DATA_PATH, TRAIN_FILE)
    TEST_FILE = 'Groceries data test.csv'
    TEST_PATH = path.join(DATA_PATH, TEST_FILE)
    MIN_SUPPORT = 0.02
    MIN_CONFIDENCE = 0.3
    MIN_LIFT = 0.1
    ALGOS = ['Apriori','FP-Growth']
    # Weights for Support, Confidence, Lift, Recency
    ALPHA = 0.2
    BETA = 0.2
    GAMMA = 0.4
    DELTA = 0.2

    df = pd.read_csv(TRAIN_PATH)

    '''
    We will first ensure that the modules function correctly in isolation.
    Then we will merge them into a single unit
    '''

    # Sample user and num_items
    n_recommended_items = 5
    user_id = 1000

    # Collaborative Filtering
    cf = CollaborativeFiltering(TRAIN_PATH)

    cf_recs = cf.get_cf_recommended_items(user_id, n_recommended_items=n_recommended_items)

    print(f'Recommendations of {n_recommended_items} items for user {user_id}:\n')
    for item, value in cf_recs:
        print(f"('{item:<14}', {value:.2f})")

    # Pattern Mining
    engine = FlexiblePatternMiner(df,
        user_col='User_id',
        item_col='itemDescription',
        date_col='Date'
    )
    engine.set_min_support(MIN_SUPPORT)
    engine.set_min_confidence(MIN_CONFIDENCE)
    engine.set_min_lift(MIN_LIFT)
    engine.set_selected_algorithms(ALGOS)
    engine.set_weights(ALPHA, BETA, GAMMA, DELTA)

    engine.mine_frequent_itemsets()
    itemsets = engine.get_frequent_itemsets()

    engine.generate_rules()

    # Apply composite scoring to the same ruleset dataframe
    engine.apply_composite_scoring()

    rules_df = engine.get_rules()

    print("Type 'exit' to quit.")
    while True:
        resp = input("Enter User ID: ").strip()
        if resp.lower() in ('exit','quit'):
            break
        try:
            uid = int(resp)
        except ValueError:
            print("→ please enter a numeric ID")
            continue

        # C.F. 
        cf_recs = cf.get_cf_recommended_items(user_id, n_recommended_items=n_recommended_items)
        print(f"\nCF-based top 5 for user {uid}:")
        for item, score in cf_recs:
            print(f" • {item:<20} score={score:.2f}")

        # — Rule-based recommendations —
        # Grab user's past items
        hist = cf.transaction_history
        purchased = hist.loc[hist['user_id']==uid, 'items'].tolist()
        purchased_items = purchased[0] if purchased else []
        rule_recs = cf.get_association_rules_recommendations(
            purchased_items, rules_df, max_recommendations=5
        )
        print(f"\nRule-based top 5 for user {uid}:")
        for item, score in rule_recs:
            print(f" • {item:<20} score={score:.2f}")

        # — Hybrid recommendations —
        hybrid = cf.weighted_hybrid_cf_recommended_items(
            rules_df, cf_recs, uid
        )
        print(f"\nHybrid top 5 for user {uid}:")
        for item, score in hybrid:
            print(f" • {item:<20} score={score:.2f}")

        print("\n" + "-"*40 + "\n")

if __name__ == '__main__':
    main()
