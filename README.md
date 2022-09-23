# Multilingual Sentiment Analysis

### Virtual Env & Requirements

Python 3.6.9
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Train on BertModel (bert-base-multilingual-cased)

```
sh train_bert.sh
```

### Train on XLMRobertaModel (xlm-roberta-large)

```
sh train_xlm_roberta.sh
```