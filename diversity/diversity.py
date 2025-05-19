
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from CFGenerators_diversity import CFGenerators

datasets = [
    'adult', 'banknote', 'biodeg', 'breast-cancer', 'compas', 'german', 'heart',
    'heloc', 'liver', 'magic', 'mammographic', 'phishing', 'pima', 'spam', 'wine'
]

methods = ['mo', 'discern', 'lire', 'eece_pparallelized']
summary_results = []

def load_dataset(name):
    data = pd.read_csv(f'data/{name}.csv')
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1]).astype(int)
    return X, y

def build_feature_constraints(X, feature_names):
    n_features = X.shape[1]
    return {
        'immutable': [],
        'increasing': [],
        'decreasing': [],
        'data types': ['int' if np.all(X[:, i] == X[:, i].astype(int)) else 'float' for i in range(n_features)]
    }

for dataset in datasets:
    X, y = load_dataset(dataset)
    feature_names = [f'f{i}' for i in range(X.shape[1])]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    method_divs = {method: {'jaccard': [], 'subset': []} for method in methods}

    def evaluate_fold(train_idx, test_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        feature_cons = build_feature_constraints(X_train, feature_names)
        cfgen = CFGenerators(rf, X_train, feature_names, feature_cons)
        cfgen.fit()

        fold_results = {method: {'jaccard': 0, 'subset': 0} for method in methods}

        for method in methods:
            jacc_divs = []
            subset_divs = []
            for i in range(min(100, len(X_test))):
                x = X_test[i]
                y_true = y_test[i]
                target = 1 - y_true if cfgen.n_classes == 2 else (y_true + 1) % cfgen.n_classes

                try:
                    cfs, _ = cfgen.generate_cf(x, target, generator=method)
                    if cfs is not None:
                        jacc_divs.append(cfgen.compute_jaccard_diversity(x, cfs))
                        subset_divs.append(cfgen.compute_diversity(x, cfs))
                except Exception:
                    continue

            fold_results[method]['jaccard'] = np.mean(jacc_divs) if jacc_divs else 0
            fold_results[method]['subset'] = np.mean(subset_divs) if subset_divs else 0

        return fold_results

    fold_results_list = Parallel(n_jobs=15)(
        delayed(evaluate_fold)(train_idx, test_idx) for train_idx, test_idx in skf.split(X, y)
    )

    for method in methods:
        method_divs[method]['jaccard'] = [res[method]['jaccard'] for res in fold_results_list]
        method_divs[method]['subset'] = [res[method]['subset'] for res in fold_results_list]

    for method in methods:
        mean_jacc = np.mean(method_divs[method]['jaccard'])
        std_jacc = np.std(method_divs[method]['jaccard'])
        mean_subset = np.mean(method_divs[method]['subset'])
        std_subset = np.std(method_divs[method]['subset'])
        summary_results.append({
            'dataset': dataset,
            'method': method,
            'jaccard_diversity': f'{mean_jacc:.4f} ± {std_jacc:.4f}',
            'subset_diversity': f'{mean_subset:.4f} ± {std_subset:.4f}'
        })

df_summary = pd.DataFrame(summary_results)
df_summary.to_csv('cf_diversity_summary.csv', index=False)
print('Summary saved to cf_diversity_summary.csv')
