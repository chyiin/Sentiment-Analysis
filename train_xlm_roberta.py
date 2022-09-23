import argparse
import torch
from torch import nn
import pandas as pd
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from XLMRobertaClassifier import XLMRoberta
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import wandb
import re

def same_seeds(seed, gpu):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'])
    parser.add_argument('--date', type=str, default='20220913') 
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--MAX_LEN', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    return args

def simple_text_clean(x):
    # remove unicode characters
    x = x.encode('ascii', 'ignore').decode()
    x = re.sub(r'https*\S+', ' ', x)
    x = re.sub(r'http*\S+', ' ', x)
    # then use regex to remove @ symbols and hashtags
    x = re.sub(r'\'\w+', '', x)
    #x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
    x = re.sub(r'\w*\d+\w*', '', x)
    x = re.sub(r'\s{2,}', ' ', x)
    x = re.sub(r'\s[^\w\s]\s', '', x)
    return x

def preprocessing_for_bert(data, lbs, MAX_LEN, labels_to_ids, tokenizer):
        
    input_ids = []
    attention_masks = []
    labels = []
    for id, sent in enumerate(tqdm(data)):
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,        
            MAX_LENgth=MAX_LEN,                  
            pad_to_MAX_LENgth=True,         
            return_attention_mask=True)
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        if lbs is None:
            labels.append(0)
        else:
            labels.append(int(labels_to_ids[lbs[id]]))

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)        

def main():

    args = parse_args()
    args = vars(args)

    train(args)

def train(args):

    DATE, GPU, SEED = args['date'],  args['gpu'], args['seed']
    LEARNING_RATE, BATCH_SIZE, EPOCHS, MAX_LEN = args['lr'], args['batch_size'], args['epoch'], args['max_len']
    
    same_seeds(SEED, GPU)

    print('\nLoading...')   

    print(f'\n[Date]: {DATE}\n[Gpu]: {GPU}\n[Seed]: {SEED}')
    print(f'[Epochs]: {EPOCHS}\n[Batch Size]: {BATCH_SIZE}\n[Learning Rate]: {LEARNING_RATE}\n[Max Length]: {MAX_LEN}\n')

    train_data = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv').sample(frac=1).reset_index(drop=True)[:100]
    valid_data = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/validation.csv').sample(frac=1).reset_index(drop=True)[:10]
    test_data = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/test.csv')

    print('Training Data Size:', train_data.shape)
    print('Validation Data Size:', valid_data.shape)
    print('Testing Data Size:', test_data.shape)

    labels_to_ids = {k: v for v, k in enumerate(sorted(list(dict.fromkeys(train_data['toxic']))))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(list(dict.fromkeys(train_data['toxic']))))}
   
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    if args['mode'] == 'train':

        if args['wandb']:
            wandb.init(project="Roberta-Sentiment-Analysis", name=f'roberta-large-{LEARNING_RATE}-{EPOCHS}-{MAX_LEN}-{BATCH_SIZE}', entity="chyiin")

        print('\nTokenizing data...\n')
        train_inputs, train_masks, train_labels = preprocessing_for_bert(train_data['comment_text'], train_data['toxic'], MAX_LEN, labels_to_ids, tokenizer)
        val_inputs, val_masks, val_labels = preprocessing_for_bert(valid_data['comment_text'], valid_data['toxic'], MAX_LEN, labels_to_ids, tokenizer)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

        validation_data = TensorDataset(val_inputs, val_masks, val_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

        model = XLMRoberta(pretrained='xlm-roberta-base', hidden_size=768, num_labels=len(labels_to_ids)).cuda()

        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer1 = list(model.named_parameters())
            no_decay1 = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters1 = [
                {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay1)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay1)],
                'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer1 = list(model.classifier.named_parameters())
            optimizer_grouped_parameters1 = [{"params": [p for n, p in param_optimizer1]}]
        
        warm_up, max_grad_norm = 0.0, 1.0
        optimizer = AdamW(optimizer_grouped_parameters1, lr=LEARNING_RATE)
        total_steps = len(train_dataloader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps*warm_up,
            num_training_steps=total_steps
        )
    
        print('\nStart Training ...')
        validation_loss_values, roc_auc_list = [], []
        for _ in range(EPOCHS):
            total_loss = 0
            print(f'\nEpoch [{_+1}/{EPOCHS}] ...')
            for step, batch in enumerate(tqdm(train_dataloader)):

                model.train()
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                model.zero_grad()
                outputs = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels) #, batch=batch)
                
                loss = outputs[0]
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            model.eval()

            eval_loss = 0
            predictions, prob_predictions, true_labels = [], [], []
            for step, batch in enumerate(tqdm(validation_dataloader)):

                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():

                    outputs = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels) #, batch=batch)

                logits = outputs[1].detach().cpu().numpy() # [:,1]
                label_ids = b_labels.to('cpu').numpy()
                eval_loss += outputs[0].item()

                sft = nn.Softmax(dim=1)
                predictions.extend(np.argmax(logits, axis=1))
                prob_predictions.extend(sft(outputs[1]).detach().cpu().numpy()[:, 1]) 
                true_labels.extend(label_ids)
           
            eval_loss = eval_loss / len(validation_dataloader)

            print(f'\nEpoch {_+1}/{EPOCHS}, [Training] Loss: {"{:.5f}".format(avg_train_loss)}, [Validation] Loss: {"{:.5f}".format(eval_loss)}')
            print(f'Epoch {_+1}/{EPOCHS}, [Validation] Label Accuracy: {"{:.5f}".format(accuracy_score(predictions, true_labels))}, ROC AUC: {"{:.5f}".format(roc_auc_score(true_labels, prob_predictions, labels=np.array([0, 1])))}')

            roc_auc_list.append(roc_auc_score(true_labels, prob_predictions))
            print()
            if roc_auc_score(true_labels, prob_predictions)>=max(roc_auc_list):
                print("saving state dict")
                torch.save(model.state_dict(), f'model_{DATE}/roberta-checkpoint-{BATCH_SIZE}-{LEARNING_RATE}-{EPOCHS}-{MAX_LEN}.pt')

            if args['wandb']:

                wandb.log({
                    'Train Loss': avg_train_loss,
                    'Validation Loss': eval_loss,
                    'Validation Accuracy': accuracy_score(predictions, true_labels),
                    'ROC AUC': roc_auc_score(true_labels, prob_predictions),
                    })

    elif args['mode'] == 'predict':

        print('\nLoading ...')
        golden = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/test_labels.csv')['toxic'].values
        test_inputs, test_masks, test_labels = preprocessing_for_bert(test_data['content'], None, MAX_LEN, labels_to_ids, tokenizer)

        # golden = abs(np.array(test_data['label'].values)-1) # pd.read_csv('jigsaw-multilingual-toxic-comment-classification/test_labels.csv')['toxic'].values
        # test_inputs, test_masks, test_labels = preprocessing_for_bert(test_data['review'], None, MAX_LEN, labels_to_ids, tokenizer)
 
        testing_data = TensorDataset(test_inputs, test_masks)
        testing_sampler = RandomSampler(testing_data)
        testing_dataloader = DataLoader(testing_data, batch_size=1, shuffle=False)

        print('\nStart Evaluation ...\n')
        model = XLMRoberta(pretrained='xlm-roberta-base', hidden_size=768, num_labels=len(labels_to_ids)).cuda()
        model.load_state_dict(torch.load(f'model_{DATE}/roberta-checkpoint-{BATCH_SIZE}-{LEARNING_RATE}-{EPOCHS}-{MAX_LEN}.pt')) 

        prob_predict, predict = [], []
        model.eval()
        for step, batch in enumerate(tqdm(testing_dataloader)):

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():

                output = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask) #, batch=batch)

            sft = nn.Softmax(dim=1)
            prob_predict.append(sft(output[1]).detach().cpu().numpy()[:, 1])
            label_indices = np.argmax(output[1].to('cpu').numpy(), axis=1)
            for label_idx in label_indices:
                predict.append(ids_to_labels[label_idx])

        print(f'\nResult on Testing Data, Accuracy: {"{:.5f}".format(accuracy_score(golden, predict))}, ROC AUC: {"{:.5f}".format(roc_auc_score(golden, prob_predict))}\n') 

if __name__ == '__main__':

    main()
