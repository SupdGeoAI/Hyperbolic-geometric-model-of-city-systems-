# Hyperbolic-geometric-model-of-city-systems

This repository provides a complete workflow for processing human mobility data, performing random walks, and training Poincaré embedding models，as well as generating a hierarchical structure of urban systems in Poincaré embedding based on central place theory.

---
## 🧩 1. Preprocessing

Prepare human mobility data into a clean and processable format, and compute daily averages.

```bash

cd ./data

python 1.data_process.py

python 2.align_city.py

python 3.model_prep.py

```

---

## 🚶 2. Random Walk Generation

Generate random walk sequences from preprocessed mobility data.

```bash

python 4.generate_seq.py

```

---
## 🧠 3. Model Training

Train the Poincaré embedding model using the generated random walk data.

```bash

cd ./train

python train_poincare.py --version <VERSION> --data_path <PATH> --lr <LR>

```

**Parameter Explanation:**

- `--version`: Model version ID (see table below)

- `--data_path`: Path to the walk data

- `--lr`: Learning rate

## 🌳 4. Tree Structure Generation 

Generate a tree structure on the trained Poincaré embedding.

```bash

cd ./tree_generation

python generate_tree_poincare.py --country CN --v <VERSION> --b <B> --crosslayer <CROSSLAYER> --k <K> 

cd ./tree_generation/exp_emb/CN_v1_embedding.pkl 
is the provided embedding result for China’s short-term 2018-19 data, used to run the tree generation code.
```

**Parameter Explanation:**
- `--version`: Model version ID (see table below)
- `--b`: The number of root nodes in the tree structure 
- `--crosslayer`: The maximum order difference between parent and child nodes
- `--k`: The parameter K used for initialization, estimated by central place theory

## 📊 Experiment Configuration

Parameters required for the experiment: Data Path and Learning Rate for model training, as well as k and crosslayer for tree generation.

| Dataset Type  | Year | Version | Data Path                                            | Learning Rate | k      | crosslayer |
| ------------- | ---- | ------- | ---------------------------------------------------- | ------------- | ------ | ---------- |
| CN Short-term | 2019 | 1       | `../data/walks_data/CN_short-term_20191018-31.walks` | 0.001         | 2.3139 | 3          |
| CN Short-term | 2020 | 2       | `../data/walks_data/CN_short-term_20201018-31.walks` | 0.001         | 2.5524 | 3          |
| CN Short-term | 2021 | 3       | `../data/walks_data/CN_short-term_20211018-31.walks` | 0.001         | 2.3381 | 3          |
| CN Long-term  | 2019 | 4       | `../data/walks_data/CN-long-term_2018-19.walk`       | 0.001         | 1.7264 | 4          |
| CN Long-term  | 2020 | 5       | `../data/walks_data/CN-long-term_2019-20.walk`       | 0.001         | 1.7499 | 4          |
| CN Long-term  | 2021 | 6       | `../data/walks_data/CN-long-term_2020-21.walk`       | 0.001         | 1.7009 | 4          |
| CN Long-term  | 2022 | 7       | `../data/walks_data/CN-long-term_2021-22.walk`       | 0.001         | 1.9975 | 4          |
| CN Long-term  | 2023 | 8       | `../data/walks_data/CN-long-term_2022-23.walk`       | 0.001         | 1.6717 | 4          |
| US Short-term | 2019 | 1       | `../data/walks_data/US-short-term_201903-0415.walk`  | 0.005         | 1.7811 | 4          |
| US Short-term | 2020 | 2       | `../data/walks_data/US-short-term_202003-0415.walk`  | 0.005         | 1.8303 | 4          |
| US Short-term | 2021 | 3       | `../data/walks_data/US-short-term_202103-0415.walk`  | 0.005         | 1.9391 | 4          |
| US Long-term  | 2019 | 4       | `../data/walks_data/US-long-term_2018-19.walk`       | 0.005         | 1.7352 | 4          |
| US Long-term  | 2020 | 5       | `../data/walks_data/US-long-term_2019-20.walk`       | 0.005         | 1.691  | 4          |
| US Long-term  | 2021 | 6       | `../data/walks_data/US-long-term_2020-21.walk`       | 0.005         | 1.781  | 4          |

---

## 🧾 Notes

- All random walks are stored in `./data/walks_data/`.

- Make sure data paths are correctly configured before running.

- For each dataset, select the corresponding `version`, `data_path`， `lr`,  `k` and `crosslayer`


