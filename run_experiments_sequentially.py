
import sys
sys.path.insert(0, ".")          # ensures both .py files are found

from data_loaders import load_ds3_neonatal
from graph_kernel_adr_prediction import run_full_pipeline

from data_loaders import (load_ds1_drug_adr_relation, load_ds2_sider,
                           load_ds3_neonatal, load_webmd, load_twosides)

configs = [
    ("DS1",      load_ds1_drug_adr_relation, {"filepath": "Dataset/Dataset 1_Drug ADR Relation.csv"}),
    ("DS2 SIDER",load_ds2_sider,             {"filepath": "Dataset/Dataset 2_SIDER 4.1_drug_adr_associations.csv", "max_triples": 30_000}),
    ("DS3 Neonatal", load_ds3_neonatal,      {"filepath": "Dataset/Dataset 3_Neonatal_Drug_ADR_Expanded.csv", "use_multidrug": True, "max_triples": 30_000}),
    ("DS4 WebMD",load_webmd,                 {"filepath": "Dataset/Dataset 4_WebMD.csv"}),
    ("TWOSIDES", load_twosides,              {"filepath": "Dataset/TWOSIDES.csv", "prr_threshold": 1.0}),
]

all_results = {}
for name, loader_fn, kwargs in configs:
    print(f"\n{'='*60}\n  Running: {name}\n{'='*60}")
    ds = loader_fn(**kwargs)
    all_results[name] = run_full_pipeline(ds, test_ratio=0.20, K_walk=3)