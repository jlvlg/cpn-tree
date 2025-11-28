#!/usr/bin/env python
# coding: utf-8

# # Configuration

# In[ ]:


import itertools

label_col = " Label"
variables = {
    "n_estimators": [10, 12, 14, 16],
    "classes": [
        ["DrDoS_DNS", "BENIGN"],
        ["DrDoS_NetBIOS", "DrDoS_SSDP", "TFTP"],
        ["DrDoS_LDAP", "DrDoS_DNS", "DrDoS_NTP", "DrDoS_MSSQL"],
    ],
    "samples": [1000, 2000, 3000, 4000],
}

scenarios = [
    {"n_estimators": estimators, "classes": classes, "samples": samples}
    for estimators, classes, samples in itertools.product(
        variables["n_estimators"], variables["classes"], variables["samples"]
    )
]


# In[ ]:


for i, scenario in enumerate(scenarios):
    print(f'Scenario {i + 1}:')
    print(f'    Classes: {', '.join(scenario['classes'])}')
    print(f'    Samples: {scenario['samples']}')
    print(f'    N Estimators: {scenario['n_estimators']}')
    print()


# # Data Preprocessing

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


benign = pd.read_csv('datasets/Benign.csv')
benign.replace([np.inf, -np.inf], np.nan, inplace=True)
benign.dropna(inplace=True)


# In[ ]:


dos = pd.read_csv("datasets/NDDoS.csv")


# In[ ]:


df = pd.concat([dos, benign], ignore_index=True)


# # Experiments

# In[ ]:


from cpn_tree.cpn_tree import CPNTree
from sklearn.ensemble import GradientBoostingClassifier
import time
from pathlib import Path
import pickle as pkl


# In[ ]:


def execute_scenario(scenario):
    path = Path(
        f"out/{'-'.join(scenario['classes'])}/{scenario['n_estimators']}_estimators/{scenario['samples']}_samples/"
    )
    path.mkdir(parents=True, exist_ok=True)

    n_classes = len(scenario["classes"])
    base_n = scenario["samples"] // n_classes
    remainder = scenario["samples"] % n_classes
    remainder_classes = np.random.RandomState(42).choice(
        scenario["classes"], remainder, replace=False
    )

    if "BENIGN" in scenario["classes"]:
        columns = [
            " Packet Length Mean",
            " Subflow Fwd Bytes",
            " Flow Packets/s",
            " Flow IAT Mean",
            " Flow Duration",
            " act_data_pkt_fwd",
            " Fwd Header Length",
            "Init_Win_bytes_forward",
            "Subflow Fwd Packets",
            " Packet Length Variance",
            " Packet Length Std",
            " ACK Flag Count",
            " min_seg_size_forward",
            " Fwd Packet Length Std",
            " Label",
        ]
    else:
        columns = [
            "Flow Bytes/s",
            "Total Length of Fwd Packets",
            "Fwd Packets/s",
            " Flow Duration",
            " Fwd Header Length",
            "Subflow Fwd Packets",
            " ACK Flag Count",
            " min_seg_size_forward",
            " Fwd IAT Min",
            " Packet Length Variance",
            " Packet Length Std",
            " Label",
        ]
    dfb = df[columns]
    sampled = [
        dfb[dfb[label_col] == class_].sample(
            base_n + (1 if class_ in remainder_classes else 0), random_state=42
        )
        for class_ in scenario["classes"]
    ]
    df_minor = pd.concat(sampled, ignore_index=True)
    print('Label distribution:')
    print(df_minor[label_col].value_counts())
    X, y = df_minor.drop(columns=[label_col]), df_minor[label_col]

    gbm = GradientBoostingClassifier(
        n_estimators=scenario["n_estimators"], random_state=42
    ).fit(X, y)
    model_predictions = pd.Series(gbm.predict(X), index=X.index)

    cpn = CPNTree().add_from_GradientBoostingClassifier("gbm", gbm)
    start_time = time.time()
    net_predictions = cpn.predict(X, write_cpn=Path(path, "net.cpn"))
    elapsed_time = time.time() - start_time

    with open(Path(path, "model.pkl"), "wb") as f:
        pkl.dump(model_predictions, f)

    with open(Path(path, "net.pkl"), "wb") as f:
        pkl.dump({**net_predictions, "elapsed_time": elapsed_time}, f)


# In[ ]:


import datetime 

for i, scenario in enumerate(scenarios):
    path = Path(
        f"out/{'-'.join(scenario['classes'])}/{scenario['n_estimators']}_estimators/{scenario['samples']}_samples/"
    )
    if not (Path(path, 'model.pkl').exists() and Path(path, 'net.pkl').exists() or Path(path, 'skipped').exists()): 
        print(f'Executing scenario {i} at {datetime.datetime.now()}: {scenario}')
        execute_scenario(scenario)

    if Path(path, 'model.pkl').exists() and Path(path, 'net.pkl').exists():
        print(f"Scenario {i} results: {scenario}")
        with open(Path(path, 'model.pkl'), 'rb') as f:
            model_predictions = pkl.load(f).apply(CPNTree.format_text)
        with open(Path(path, 'net.pkl'), "rb") as f:
            net = pkl.load(f)
            net_predictions = net['gbm']
            elapsed_time = net['elapsed_time']
        comparisons = model_predictions == net_predictions
        print(f"Matches: {len(comparisons[comparisons])}/{len(comparisons)} -> {comparisons.mean() * 100}%/100%")
        print(f"Elapsed time: {elapsed_time} seconds / {elapsed_time/60} minutes / {elapsed_time/3600} hours")
        print()

