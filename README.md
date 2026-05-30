# Pone-GNN

✨ **Pone-GNN** is a graph neural recommender that learns from both **positive** and **negative** feedback.

It is the codebase for the paper **Pone-GNN: Integrating Positive and Negative Feedback in Graph Neural Networks for Recommender Systems**.

[paper](https://dl.acm.org/doi/full/10.1145/3711666)

## 🔍 Overview

Most GNN based recommenders only learn from what users like.  
Pone-GNN also learns from what users dislike.

It does this with:

- two embeddings for each user and item
  - **interest embedding**
  - **disinterest embedding**
- separate message passing on the positive graph and negative graph
- contrastive learning across the two feedback views
- a disinterest score filter for final ranking

## 🧠 Key Idea

Pone-GNN follows a simple pipeline:

1. Split interactions into a positive graph and a negative graph
2. Learn interest embeddings from positive feedback
3. Learn disinterest embeddings from negative feedback
4. Filter out high disinterest items before final recommendation

This helps reduce irrelevant recommendations and improves ranking quality.

## 📈 Results

The paper reports strong results on **ML-1M**, **Amazon-Book**, **Yelp**, and **KuaiRec**.

Highlighted result:

- **+6.15% relative nDCG@10** over the runner up on **KuaiRec**

Selected `nDCG@10` from the paper:

| Dataset | nDCG@10 |
| --- | ---: |
| ML-1M | 38.16 |
| Amazon-Book | 9.31 |
| Yelp | 6.25 |
| KuaiRec | 43.99 |

## 📁 Repository Structure

```text
.
├── Pone-GNN-main/
│   ├── ponegnn.py
│   ├── convols.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── data_loader.py
│   ├── util.py
│   ├── run.sh
│   └── ml-1m/
```

## ⚙️ Requirements

- Python 3.9+
- PyTorch
- PyTorch Geometric
- pandas
- numpy
- tqdm

Example install:

```bash
pip install torch pandas numpy tqdm
pip install torch-geometric
```

## 🚀 Quick Start

```bash
cd Pone-GNN-main
python trainer.py --dataset ML-1M --version 1 --aggregate ponegnn --K 40
```

Or run:

```bash
bash run.sh
```

## 🗂️ Data

This repo snapshot currently includes **ML-1M** files:

- `ratings.dat`
- `train_1m1.dat`
- `test_1m1.dat`

For explicit ratings, the paper uses **3.5** as the feedback threshold:

- `rating > 3.5`: positive feedback
- `rating < 3.5`: negative feedback

## 📌 Citation

```bibtex
@article{liu2025ponegnn,
  title={Pone-GNN: Integrating Positive and Negative Feedback in Graph Neural Networks for Recommender Systems},
  author={Liu, Ziyang and Wang, Chaokun and Zheng, Shuwen and Wu, Cheng and Zheng, Kai and Song, Yang and Mou, Na},
  journal={ACM Transactions on Recommender Systems},
  volume={3},
  number={2},
  pages={1--23},
  year={2025},
  doi={10.1145/3711666}
}
```
