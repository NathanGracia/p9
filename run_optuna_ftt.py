import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks"))

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import rtdl
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 314
N_TRIALS = 300
N_FOLDS = 3
TARGET = 'result'

CONTEXT = ['side', 'firstblood']
FEATS_10 = CONTEXT + ['golddiffat10','xpdiffat10','csdiffat10','killsat10','assistsat10','deathsat10']
FEATS_15 = FEATS_10 + ['firstdragon','firstherald','firsttower','golddiffat15','xpdiffat15','csdiffat15','killsat15','assistsat15','deathsat15']
FEATS_20 = FEATS_15 + ['golddiffat20','xpdiffat20','csdiffat20','killsat20','assistsat20','deathsat20']
FEATS_25 = FEATS_20 + ['firstbaron','golddiffat25','xpdiffat25','csdiffat25','killsat25','assistsat25','deathsat25']
CHECKPOINTS = {'10min': FEATS_10, '15min': FEATS_15, '20min': FEATS_20, '25min': FEATS_25}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')

df = pd.read_csv('../data/data_cleaned.csv')
print(f'Dataset : {len(df)} lignes\n')


def make_cv_objective(X_train, y_train, n_features):
    def objective(trial):
        d_token    = trial.suggest_categorical('d_token', [64, 128, 192, 256, 320, 512])
        n_blocks   = trial.suggest_int('n_blocks', 1, 6)
        ffn_factor = trial.suggest_float('ffn_factor', 0.5, 4.0)
        ffn_d_hidden = max(int(d_token * ffn_factor) // 2 * 2, 4)
        attn_drop  = trial.suggest_float('attention_dropout', 0.0, 0.4)
        ffn_drop   = trial.suggest_float('ffn_dropout', 0.0, 0.4)
        res_drop   = trial.suggest_float('residual_dropout', 0.0, 0.3)
        lr         = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_aucs = []

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            scaler = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            try:
                model = rtdl.FTTransformer.make_baseline(
                    n_num_features=n_features, cat_cardinalities=None,
                    d_token=d_token, n_blocks=n_blocks,
                    attention_dropout=attn_drop,
                    ffn_d_hidden=ffn_d_hidden, ffn_dropout=ffn_drop,
                    residual_dropout=res_drop,
                    d_out=1,
                ).to(device)
            except Exception:
                return 0.0

            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            X_t = torch.FloatTensor(X_tr_s).to(device)
            y_t = torch.FloatTensor(y_tr).to(device)
            X_v = torch.FloatTensor(X_val_s).to(device)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

            best_auc, patience_ctr = 0.0, 0
            for _ in range(150):
                model.train()
                for xb, yb in loader:
                    opt.zero_grad()
                    F.binary_cross_entropy_with_logits(model(xb, None).squeeze(1), yb).backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    probas = torch.sigmoid(model(X_v, None).squeeze(1)).cpu().numpy()
                auc = roc_auc_score(y_val, probas)
                if auc > best_auc:
                    best_auc, patience_ctr = auc, 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= 15:
                        break

            fold_aucs.append(best_auc)

        return float(np.mean(fold_aucs))
    return objective


best_params = {}

for name, feats in CHECKPOINTS.items():
    print(f'{"="*55}')
    print(f'Optuna FTT @{name} — {N_TRIALS} trials, {N_FOLDS}-fold CV')
    print(f'{"="*55}')

    subset = df[feats + [TARGET]].dropna()
    X = subset[feats].values
    y = subset[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    sampler = optuna.samplers.CmaEsSampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        make_cv_objective(X_train, y_train, len(feats)),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params[name] = study.best_params
    print(f'  Meilleure AUC CV : {study.best_value:.4f}')
    print(f'  Params : {study.best_params}\n')


print('Re-entrainement final sur train complet avec best params...\n')

from sklearn.metrics import accuracy_score, f1_score

final_results = {}

for name, feats in CHECKPOINTS.items():
    print(f'@{name}')
    subset = df[feats + [TARGET]].dropna()
    X = subset[feats].values
    y = subset[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    p = best_params[name]
    d_token      = p['d_token']
    ffn_d_hidden = max(int(d_token * p['ffn_factor']) // 2 * 2, 4)

    model = rtdl.FTTransformer.make_baseline(
        n_num_features=len(feats), cat_cardinalities=None,
        d_token=d_token, n_blocks=p['n_blocks'],
        attention_dropout=p['attention_dropout'],
        ffn_d_hidden=ffn_d_hidden, ffn_dropout=p['ffn_dropout'],
        residual_dropout=p['residual_dropout'],
        d_out=1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=1e-5)
    X_tr = torch.FloatTensor(X_train_s).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test_s).to(device)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=p['batch_size'], shuffle=True)

    best_acc, patience_ctr, best_state = 0, 0, None
    for _ in range(200):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(xb, None).squeeze(1), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = accuracy_score(y_test, (model(X_te, None).squeeze(1) > 0).cpu().numpy())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 20:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probas = torch.sigmoid(model(X_te, None).squeeze(1)).cpu().numpy()
    preds = (probas > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)
    f1  = f1_score(y_test, preds)
    print(f'  Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}')

    final_results[name] = {'model': model.cpu(), 'scaler': scaler, 'acc': acc, 'auc': auc}

    tag = name.replace(' ', '')
    torch.save(model.cpu(), f'../models/ftt/ftt_{tag}.pt')
    with open(f'../models/preprocessing/scaler_{tag}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

with open('../models/ftt_opt_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print('\nTermine. Recap final :')
for name, r in final_results.items():
    print(f'  @{name} -> Acc: {r["acc"]:.4f} | AUC: {r["auc"]:.4f}')
