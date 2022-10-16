# ComFA-CL
Source code for model ConvE-CFR
## Installation

This repo supports Linux and Python 3.x.

Install the requirements: `pip install -r requirements.txt`

# Running the model:
WN18RR:
```
python comfacl.py --model conve --data WN18RR --preprocess --CFR_kernels 32 --lr 0.001
```

FB15k-237:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data WN18RR --preprocess --CFR_kernels 32 --lr 0.001
```

YAGO3-10:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data YAGO3-10 --preprocess --lr 0.001
```

NELL-995:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data NELL-995 --preprocess --lr 0.001 --test-batch-size 64 --batch-size 64
```

# Acknowledgements
The code is inspired by [InteractE](https://github.com/malllabiisc/InteractE).
