import chunk
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
import re

def get_chunks(s, maxlength):
    start = 0
    end = 0
    while start + maxlength  < len(s) and end != -1:
        end = s.rfind(" ", start, start + maxlength + 1)
        yield s[start:end]
        start = end +1
    yield s[start:]

def chunk_text(data, colnames, type):

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    chunk_sent, chunk_id, chunk_toxic = [], [], []
    for id, sent in enumerate(tqdm(data[colnames])):
        chunks = get_chunks(re.sub(' +', ' ', sent.strip()), 100)
        for ck in chunks:
            chunk_sent.append(ck)
            chunk_id.append(data['id'][id])
            if type == 'test':
                chunk_toxic.append(0)
            else:
                chunk_toxic.append(data['toxic'][id])
    df = pd.DataFrame({'id': chunk_id, 'comment_text': chunk_sent, 'toxic': chunk_toxic}) # .dropna().reset_index(drop=True)
    chunk_df = df[df['comment_text'].notna()]
    print(chunk_df.shape)
    chunk_df.to_csv(f'chunk_dataset/{type}_chunk.csv', index=False)

train = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv') # .sample(frac=1).reset_index(drop=True)[:100]
valid = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/validation.csv') # .sample(frac=1).reset_index(drop=True)[:10]
test = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/test.csv')

chunk_text(train, 'comment_text', 'train') # 990685
chunk_text(valid, 'comment_text', 'valid') # 33679
chunk_text(test, 'content', 'test') # 274677

# print(chunk_df.groupby(['id']).mean().head())
# print(chunk_df.groupby(['id']).mean().shape)
