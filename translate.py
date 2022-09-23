
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm
import re

train_data = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

transl_text = []
for src_text in tqdm(train_data['comment_text']):
    translated = model.generate(**tokenizer(re.sub(' +', ' ', src_text.strip()), return_tensors="pt", padding=True))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    transl_text.append(tgt_text[0])

raw_data = {'id': train_data['id'].values,
            'comment_text': transl_text,
            'toxic': train_data['toxic'].values}

df = pd.DataFrame(raw_data, columns = ['id', 'comment_text', 'toxic'])
print(df.head())
df.to_csv('train_translate.csv')
