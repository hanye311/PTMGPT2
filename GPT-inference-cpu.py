#Import necessary libraries
import os
import torch
# import evaluate
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import DataCollatorWithPadding,DataCollatorForSeq2Seq
from transformers import AutoTokenizer, GPT2LMHeadModel,TrainingArguments, Trainer,GPT2Config,EarlyStoppingCallback
from sklearn.metrics import average_precision_score,matthews_corrcoef,f1_score, precision_score, recall_score, balanced_accuracy_score
from tqdm import tqdm
#Use GPU else specify '-1' for CPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Map positive/negative labels and prepare prompt for inference
class SequenceClassificationDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.map_label={0:'NEGATIVE',1:'POSITIVE'}
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        prep_txt= f'<startoftext>SEQUENCE:{sequence}\nLABEL:'
        encoding = self.tokenizer(prep_txt,return_tensors='pt',padding='longest')
        return  {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label':label
            # 'labels':encoding['input_ids'].squeeze()

        }
    # def __getitem__(self, idx):
    #     sequence = self.sequences[idx]
    #     label = self.labels[idx]
    #     prep_txt1= f'<startoftext>SEQUENCE:{sequence}\nLABEL:{self.map_label[label]}<endoftext>'
    #     encoding1 = self.tokenizer(prep_txt1,return_tensors='pt')
    #     return {
    #         'input_ids': encoding1['input_ids'].squeeze(),
    #         'attention_mask': encoding1['attention_mask'].squeeze(),
    #         'labels': encoding1['input_ids'].squeeze()
    #     }

def get_score(mdl_path,test_data_loader):
    model_config = GPT2Config.from_pretrained(mdl_path)
    model = GPT2LMHeadModel.from_pretrained(mdl_path,config=model_config,ignore_mismatched_sizes=True)
    # model=model.cuda().eval()
    model = model.cpu().eval()
    predition=[]
    for i,x in tqdm(enumerate(test_data_loader)):

        Actual=f"{tokenizer.decode(x['input_ids'][0],skip_special_tokens=True)} {x['label']}"
        # generated=x['input_ids'].cuda()
        generated = x['input_ids']
        # sample_outputs = model.generate(generated, attention_mask=x['attention_mask'].cuda(), do_sample=False, top_k=50,
        #                                 max_new_tokens=2, top_p=0.15, temperature=0.1, num_return_sequences=1,
        #                                 pad_token_id=tokenizer.eos_token_id)

        # sample_outputs=model.generate(generated,attention_mask=x['attention_mask'].cuda(),do_sample=False,top_k=50,max_new_tokens=2,top_p=0.15,temperature=0.1,num_return_sequences=0,pad_token_id=tokenizer.eos_token_id)
        #
        sample_outputs = model.generate(generated, attention_mask=x['attention_mask'], do_sample=False, top_k=50,
                                        max_new_tokens=2, top_p=0.15, temperature=0.1, num_return_sequences=0,
                                        pad_token_id=tokenizer.eos_token_id)

        predicted_text=tokenizer.decode(sample_outputs[0],skip_special_tokens=True)
        print("Actual:",Actual)
        print("predicted_text:",predicted_text)
        predicted_text.split('LABEL:')[-1]
        predition+=[[map_label[int(x.pop('label'))],predicted_text.split('LABEL:')[-1]]]
    labels=[[0 if y=='NEGATIVE' else 1  for y in x] for x in predition]
    labels=np.asanyarray(labels)
    actual=labels[:,0]
    pred=labels[:,1]

    # dictionary of lists
    dict = {'label': list(actual), 'prediction': list(pred)}

    df = pd.DataFrame(dict)


    return df,f1_score(actual,pred),matthews_corrcoef(actual,pred), precision_score(actual,pred), recall_score(actual, pred), average_precision_score(actual,pred)

if __name__ == "__main__":

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2', bos_token='<startoftext>', eos_token='<endoftext>',
                                              pad_token='<PAD>')

    # Add custom tokens
    tokenizer.add_tokens(['SEQUENCE:', 'LABEL:', 'POSITIVE', 'NEGATIVE'])
    input_folder = "uniprot_dataset/"

    # ptm_list=['ubiquitylation','phosphorylation','acetylation','n_glycosylation','o_glycosylation','methylation']

    balanced = False
    ptm_list = ["Phosphoserine_Metazoa", "Phosphotyrosine_Metazoa"]
    for ptm in ptm_list:
        if balanced == True:
            ptm_file = "GPT-dataset/different_species/" + ptm + '/test_balanced.csv'
        else:
            ptm_file = input_folder + ptm + '_21nt.csv'
        # Load benchmark dataset
        data = pd.read_csv(ptm_file)
        data['Seq'] = data['Seq'].str.replace('\n', '')
        data['Seq'] = data['Seq'].str.replace('-', '')

        print(data['Label'].value_counts())

        test_texts = data['Seq'].reset_index(drop=True)
        test_labels = data['Label'].reset_index(drop=True)

        test_dataset = SequenceClassificationDataset(test_texts, test_labels, tokenizer)
        test_data_loader = DataLoader(test_dataset,
                                      collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest'),
                                      batch_size=1)

        # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest'),

        map_label = {0: 'NEGATIVE', 1: 'POSITIVE'}

        # replace the path with best performing checkpoint
        # folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/"
        if ptm=="Phosphoserine_Metazoa":
            folder = "PTMGPT2-models-Part2/Phosphorylation (S,T)/"
        elif ptm=="Phosphotyrosine_Metazoa":
            folder="PTMGPT2-models-Part2/Phosphorylation (Y)/"
        file = folder
        df, f1, mcc, prc, rec, avg = get_score(file, test_data_loader)
        print(file)
        print("f1,mcc,precision,recall,avg:", f1, mcc, prc, rec, avg)
        with open(input_folder + 'test_results-Prompt1.csv', 'a') as f:
            f.write(f'{ptm},{f1},{mcc},{prc},{rec},{avg}\n')

        if balanced == True:
            df.to_csv(file + "/test_balanced_result.csv", index=False)
        else:
            df.to_csv(input_folder + "test_result.csv", index=False)

    # # results = []
    # folder="/mnt/pixstor/xudong-lab/yehan/results-Prompt1/"
    # # Replace the path with the output directory used during model training
    # for mdl in os.listdir(folder):
    #     if 'checkpoint' in mdl:
    #         mdl_path =folder + mdl
    #         print(mdl_path)
    #         f1, mcc, prc, rec, avg = get_score(mdl_path,test_data_loader)
    #         print("f1, mcc, prc, rec, avg:",f1, mcc, prc, rec, avg)
    #         with open(folder+'val_results-Prompt1.csv', 'a') as f:
    #             f.write(f'{mdl},{f1},{mcc},{prc},{rec},{avg}\n')





