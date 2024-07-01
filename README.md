# Attention-based Causal Representation Learning for Out-of-Distribution Recommendation

## Environment
- Anaconda 3
- python 3.8.10
- pytorch 2.3.0
- numpy 1.24.3

## Data
The experimental data are in './data' folder, including, Meituan and Yelp. Due to the large size, 'item_feature.npy' of Yelp is uploaded to [Google drive](https://drive.google.com/drive/folders/1nKk15UlYzGVKCo5yMFVmW4yewbwid0dH?usp=sharing).

## Training
```
python main.py --model_name=$1 --dataset=$2 --mlp_dims=$3 --mlp_p1_1_dims=$4 --mlp_p1_2_dims=$5 --mlp_p2_dims=$6 --mlp_p3_dims=$7 --lr=$8 --wd=$9 --batch_size=$10 --epochs=$11 --total_anneal_steps=$12 --anneal_cap=$13 --CI=$14 --dropout=$15 --Z1_hidden_size=$16 --E2_hidden_size=$17 --Z2_hidden_size=$18 --bn=$19 --sample_freq=$20 --regs=$21 --act_function=$22 --log_name=$23 --gpu=$24 --cuda
```
- The explanation of hyper-parameters can be found in './main.py'. See `python main.py -h` for more information.
- The default hyper-parameter settings are detailed in './hyper-parameters.txt'.

## Examples

Train ABCORV on iid meituan:

```
python main.py --model_name=ABCORV --dataset=meituan --mlp_dims=[3500] --mlp_p1_1_dims=[] --mlp_p1_2_dims=[1] --mlp_p2_dims=[] --mlp_p3_dims=[] --lr=1e-3 --wd=0 --batch_size=500 --epochs=300 --total_anneal_steps=0 --anneal_cap=0.1 --CI=1 --dropout=0.6 --Z1_hidden_size=500 --E2_hidden_size=2000 --Z2_hidden_size=500 --bn=0 --sample_freq=1 --regs=0 --act_function=tanh --log_name=log --gpu=0 --cuda
```



- Some sources from  [Causal Representation Learning for Out-of-Distribution Recommendation](https://github.com/Linxyhaha/COR) is partially used.

