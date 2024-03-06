#  NOTE
Due to certain bugs in the evaluation metric (AP) used in the original experiment, the results for visual sarcasm target identification in the paper tend to be overly idealized. Therefore, we kindly request that researchers refrain from referencing the experimental results related to visual irony target identification in the paper.
More explanation is coming soon!

# Multimodal Sarcasm Target Identification in Tweets

## Environment
### Python packages
>- python==3.7
>- torch==1.8.0
>- gensim==3.8.0 
>- numpy==1.18.3
>- torchcrf==1.0.4
>- pytorch-pretrained-bert==0.6.2
**(pip install -r requirements.txt)**

### Configuration
All configuration are listed in main.py. Please verify parameters before running the codes.

### Data

```
datasets/
├── images/ 
├── Visual target labels/
└── Textual target labels/
    ├──train
    ├──val
    └──test
```


####The MSTI datasets format is as follows:

```
IMGID:9737
nice	O
warm	O
running	O
weather	B-S
this	O
morning	O
...	O

IMGID:9516
andy	B-S
murray	I-S
thrown	O
out	O
of	O
#	O
australianopen	O
after	O
celebrating	O
his	O
win	O
.	O
```

### Pre-trained Models
```
pretrained/
├── bert-base-uncased/
│   ├── vocab.txt
│   ├── bert_config.json
│   └── pytorch_model.bin
├── bert-large-uncased/
│   ├── vocab.txt
│   ├── bert_config.json
│   └── pytorch_model.bin
└── yolo/
```
## Usage
### Training
>- python main.py 

### Testing
>- python test.py
