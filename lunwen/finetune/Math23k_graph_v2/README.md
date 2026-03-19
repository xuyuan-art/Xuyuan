# MWP-BERT on Chinese Dataset

## Network Training

**1. How to train the network on Math23k dataset only:**
```
python math23k.py
```

**Graph-enhanced training (new experimental options):**
```
python math23k_graph.py \
  --save_dir models_graph_v2 \
  --graph_layers 2 \
  --graph_active_relations all \
  --graph_log_every 100
```
默认评估节奏是 `1/5/10/15/...`（`epoch=1` 必评估，之后每 `5` 个评估一次，`--eval_every 5`），并且只在达到 `best` 时覆盖写入 `save_dir` 根目录的模型权重，同时更新 `best_metrics.json`。
按当前脚本默认，`epoch checkpoint` 每 `10` 个 epoch 保存一次（`--epoch_ckpt_every 10`）。

**Quick ablation: disable graph branch**
```
python math23k_graph.py --disable_graph 1 --save_dir models_graph_ablation
```

**2. How to train the network on Math23k dataset and Ape-clean jointly:**
```
python math23k_wape.py
```
## Weight Loading

To load the pre-trained MWP-BERT model or other pre-trained models in Huggingface, there are two lines of code need changing:

**1. src/models.py, Line 232:**
```
self.bert_rnn = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
```
Load the model from your desired path.

**2. src/models.py, Line 803/903/1039:**
```
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
```
Load the tokenizer from your backbone model.

## MWP-BERT weights

Please find at https://drive.google.com/drive/folders/1QC7b6dnUSbHLJQHJQNwecPNiQQoBFu8T?usp=sharing.

## Citation

```
@inproceedings{liang2022mwp,
  title={MWP-BERT: Numeracy-Augmented Pre-training for Math Word Problem Solving},
  author={Liang, Zhenwen and Zhang, Jipeng and Wang, Lei and Qin, Wei and Lan, Yunshi and Shao, Jie and Zhang, Xiangliang},
  booktitle={Findings of NAACL 2022},
  pages={997--1009},
  year={2022}
}
```
