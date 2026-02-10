import os
import torch
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import DataCollatorWithPadding,DataCollatorForSeq2Seq
from transformers import AutoTokenizer, GPT2LMHeadModel,TrainingArguments, Trainer,GPT2Config,EarlyStoppingCallback

#Use GPU else specify '-1' for CPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Function to extract parts of the sequence

def extract_seq_parts(seq):
    first_ten = seq[:10]  # First ten characters
    last_ten = seq[-10:]  # Last ten characters
    center = seq[len(seq) // 2] if len(seq) % 2 != 0 else ''  # Middle character for odd length, empty for even
    return first_ten, center, last_ten

#Map positive/negative labels and prepare prompt for training
class SequenceClassificationDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer,dtype='Train'):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.map_label={1:'POSITIVE',0:'NEGATIVE'}
        self.dtype='Train'
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        prep_txt1= f'<startoftext>SEQUENCE:{sequence}\nLABEL:{self.map_label[label]}<endoftext>'
        encoding1 = self.tokenizer(prep_txt1,return_tensors='pt')
        return {
            'input_ids': encoding1['input_ids'].squeeze(),
            'attention_mask': encoding1['attention_mask'].squeeze(),
            'labels': encoding1['input_ids'].squeeze()
        }

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
   # Calculate precision, recall, and F1-score
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,    #Check the positive and negative labels
        'f1': f1
    }


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2', bos_token='<startoftext>', eos_token='<endoftext>',
                                              pad_token='<PAD>')
    # Initialize tokenizer
    # Add custom tokens
    tokenizer.add_tokens(['SEQUENCE:', 'LABEL:', 'POSITIVE', 'NEGATIVE'])

    print(tokenizer.special_tokens_map)


    data = pd.read_csv('GPT-dataset/n_glycosylation/train_remove_rebundancy_0.9.csv') # Remove \n and - characters from the sequence
    data['Seq'] = data['Seq'].str.replace('\n', '')
    data['Seq'] = data['Seq'].str.replace('-', '')
    print(data['Label'].value_counts())

    train_texts = data['Seq'].reset_index(drop=True)
    train_labels = data['Label'].reset_index(drop=True)
    train_dataset = SequenceClassificationDataset(train_texts, train_labels, tokenizer, 'Train')

    # data = pd.read_csv('GPT-dataset/acetylation/val.csv') # Remove \n and - characters from the sequence
    # data['Seq'] = data['Seq'].str.replace('\n', '')
    # data['Seq'] = data['Seq'].str.replace('-', '')
    # print(data['Label'].value_counts())
    #
    # val_texts = data['Seq'].reset_index(drop=True)
    # val_labels = data['Label'].reset_index(drop=True)
    # val_dataset = SequenceClassificationDataset(val_texts, val_labels, tokenizer)

    # Load the pre-trained model
    model_config = GPT2Config.from_pretrained('nferruz/ProtGPT2')

    training_args = TrainingArguments(
        output_dir='/mnt/pixstor/xudong-lab/yehan/results-Prompt1/n_glycosylation/',  # output directory
        num_train_epochs=200,  # total number of training epochs  #todo
        # eval_steps=500,
        save_steps=1500,
        logging_steps=500,
        logging_dir='/mnt/pixstor/xudong-lab/yehan/logs/n_glycosylation/',

        per_device_train_batch_size=128,  # batch size per device during training
        per_device_eval_batch_size=128,  # batch size for evaluation

        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        # save_total_limit = 2,  # no. of models to save in the output directory
        # load_best_model_at_end = True,
        learning_rate=5E-05,
        # seed=0,
        # metric_for_best_model='f1',
        # evaluation_strategy='epoch',
        # save_strategy='epoch'
        # greater_is_better=True,

        # report_to="tensorboard"

    )

    model = GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2', config=model_config, ignore_mismatched_sizes=True)
    print(model.resize_token_embeddings(len(tokenizer)))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest'),
        # compute_metrics=compute_metrics
    )

    trainer.train()

