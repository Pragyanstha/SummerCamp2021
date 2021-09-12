# SummerCamp2021 Team C
## セットアップ
```
git clone https://github.com/Pragyanstha/SummerCamp2021.git
cd SummerCamp2021
conda create -n "sumcamp"
conda activate sumcamp
conda install -c conda-forge -c pytorch --file requirements.txt
```

## 方針
TransGAN[https://arxiv.org/abs/2102.07074] をつかう
### 動作確認
CIFAR10で学習  
```
python exps/cifar_train.py
```
### Tensorboardで学習を確認（別のターミナルで実行してね）
SummerCamp2021内にいることを確認して
```
tensorboard serve --logdir .
```
## エラーがでた場合
エラー文をGithubのissueを作ってね