# ComFA-CL
Source code for model ComFA-CL(unsup).

## Installation

This repo supports Linux and Python 3.x.

Install the requirements: `pip install -r requirements.txt`

## Running the model:
WN18RR:
```
python comfacl.py --data WN18RR --train_strategy one_to_n --feat_drop 0.1 --hid_drop 0.6 --perm 1 --ker_sz 5 --num_filt 112 --gpu 0 --lr 0.003 --batch 128 --temperature 0.05 --lamb 0.00005 --topK 10 --ratio 0.50
```

FB15k-237:
```
python comfacl.py --data FB15k-237 --gpu 0 --temperature 0.05 --lamb 0.00005 --topK 20 --ratio 1.0
```

YAGO3-10:
```
python comfacl.py --data YAGO3-10 --train_strategy one_to_n --inp_drop 0.2 --feat_drop 0.2 --hid_drop 0.3 --ker_sz 7 --num_filt 64 --perm 2 --lr 0.001 --temperature 0.05 --lamb 0.00005 --topK 10 --ratio 1.0
```

NELL-995:
```
python comfacl.py --data NELL-995 --train_strategy one_to_n --feat_drop 0.1 --hid_drop 0.3 --ker_sz 11 --num_filt 64 --perm 4 --lr 0.001 --temperature 0.05 --lamb 0.0001 --topK 5 --ratio 0.75
```

## Acknowledgements
The code is inspired by [InteractE](https://github.com/malllabiisc/InteractE).
