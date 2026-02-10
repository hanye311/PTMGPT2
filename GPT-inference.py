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
from sklearn.metrics import average_precision_score,accuracy_score, matthews_corrcoef,f1_score, precision_score, recall_score, balanced_accuracy_score
from tqdm import tqdm
import json
from sklearn.metrics import precision_recall_curve, auc,roc_auc_score,precision_score, recall_score,accuracy_score,f1_score,matthews_corrcoef

#Use GPU else specify '-1' for CPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Map positive/negative labels and prepare prompt for inference
class SequenceClassificationDataset(Dataset):
    def __init__(self, sequences, labels,tokenizer):
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
            'true_label':label,
            'labels':encoding['input_ids'].squeeze(),


        }

def get_score(mdl_path,test_data_loader):
    model_config = GPT2Config.from_pretrained(mdl_path)
    model = GPT2LMHeadModel.from_pretrained(mdl_path,config=model_config,ignore_mismatched_sizes=True)
    model=model.cuda().eval()
    # model = model.cpu().eval()
    predition=[]
    uniprot_id_list=[]
    for i,x in tqdm(enumerate(test_data_loader)):
        Actual=f"{tokenizer.decode(x['input_ids'][0],skip_special_tokens=True)} {x['true_label']}"
        generated=x['input_ids'].cuda()
        # generated = x['input_ids']
        sample_outputs = model.generate(generated, attention_mask=x['attention_mask'].cuda(), do_sample=False, top_k=50,
                                        max_new_tokens=2, top_p=0.15, temperature=0.1, num_return_sequences=1,
                                        pad_token_id=tokenizer.eos_token_id)

        # sample_outputs=model.generate(generated,attention_mask=x['attention_mask'].cuda(),do_sample=False,top_k=50,max_new_tokens=2,top_p=0.15,temperature=0.1,num_return_sequences=0,pad_token_id=tokenizer.eos_token_id)
        #
        # sample_outputs = model.generate(generated, attention_mask=x['attention_mask'], do_sample=False, top_k=50,
        #                                 max_new_tokens=2, top_p=0.15, temperature=0.1, num_return_sequences=0,
        #                                 pad_token_id=tokenizer.eos_token_id)

        predicted_text=tokenizer.decode(sample_outputs[0],skip_special_tokens=True)
        # print("Actual:",Actual)
        # print("predicted_text:",predicted_text)
        predicted_text.split('LABEL:')[-1]
        predition+=[[map_label[int(x.pop('true_label'))],predicted_text.split('LABEL:')[-1]]]


    labels=[[0 if y=='NEGATIVE' else 1  for y in x] for x in predition]
    labels=np.asanyarray(labels)
    actual=labels[:,0]
    pred=labels[:,1]

    # dictionary of lists
    dict = {'label': list(actual), 'prediction': list(pred)}

    df = pd.DataFrame(dict)

    return df, f1_score(actual,pred),accuracy_score(actual,pred),matthews_corrcoef(actual,pred), precision_score(actual,pred), recall_score(actual, pred), average_precision_score(actual,pred)

if __name__ == "__main__":

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2', bos_token='<startoftext>', eos_token='<endoftext>',
                                              pad_token='<PAD>')

    # Add custom tokens
    tokenizer.add_tokens(['SEQUENCE:', 'LABEL:', 'POSITIVE', 'NEGATIVE'])
    input_folder = "/mnt/pixstor/data/yhhdb/PTMGPT/ptmgpt2_result/2_test_21nt_peptides/"
    output_folder= "/mnt/pixstor/data/yhhdb/PTMGPT/ptmgpt2_result/3_test_output_results/"

    # ptm_list=['ubiquitylation','phosphorylation','acetylation','n_glycosylation','o_glycosylation','methylation']

    balanced = False
    # ptm_list = ["Phosphoserine_Metazoa", "Phosphotyrosine_Metazoa"]
    # ptm_list= [
    #      "Acetylation_K",
    #     #            "Amidation_V",
    #     #            "Hydroxylation_K",
    #     #            "Hydroxylation_P",
    #                "Methylation_K",
    #                "Methylation_R",
    #                "N-linked_Glycosylation_N",
    #                "O-linked_Glycosylation_S",
    #                "O-linked_Glycosylation_T",
    #                "Palmitoylation_C",
    #             "Phosphorylation_S",
    #             "Phosphorylation_T",
    #             "Phosphorylation_Y",
    #                # "S-nitrosocysteine_C",
    #                "Succinylation_K",
    #                "Sumoylation_K",
    #                 "Ubiquitination_K"
    # ]
    # ptm_list=[
    #      "acetylation",
    #      "methylation",
    #     "n_glycosylation",
    #     "o_glycosylation",
    #     "phosphorylation",
    #     "succinyllysine",
    #     "ubiquitylation"
    #         ]
    ptm_list = [
                           "Acetylation_K",
                           "Methylation_K",
                           "Methylation_R",
                           "NlinkedGlycosylation_N",
                           "OlinkedGlycosylation_S",
                           "OlinkedGlycosylation_T",
                           "Palmitoylation_C",
                            "Phosphorylation_S",
                            "Phosphorylation_T",
                            "Phosphorylation_Y",
                           "Succinylation_K",
                           "Sumoylation_K",
                            "Ubiquitination_K"
        ]
    for ptm in ptm_list:
        print(ptm+"is predicting")
        if balanced == True:
            ptm_file = "GPT-dataset/different_species/" + ptm + '/test_balanced.csv'
        else:
            ptm_file = os.path.join(input_folder,ptm + '_21nt.csv')
        # Load benchmark dataset
        data = pd.read_csv(ptm_file)
        data['Seq'] = data['Seq'].str.replace('\n', '')
        data['Seq'] = data['Seq'].str.replace('-', '')

        print(data['Label'].value_counts())

        test_texts = data['Seq'].reset_index(drop=True)
        test_labels = data['Label'].reset_index(drop=True)


        test_dataset = SequenceClassificationDataset(test_texts, test_labels,tokenizer)
        test_data_loader = DataLoader(test_dataset,
                                      collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest'),
                                      batch_size=1)

        # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest'),

        map_label = {0: 'NEGATIVE', 1: 'POSITIVE'}

        # replace the path with best performing checkpoint
        # folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/"
        if ptm == "Phosphorylation_S" or ptm == "phosphorylation":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Phosphorylation (S,T)/"
        elif ptm == "Phosphorylation_Y":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Phosphorylation (Y)/"
        elif ptm == "Phosphorylation_T":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Phosphorylation (S,T)/"
        elif ptm == "Sumoylation_K":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Sumoylation (K)/"
        elif ptm == "Ubiquitination_K" or ptm == "ubiquitylation":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Ubiquitination (K)/"
        elif ptm == "Acetylation_K" or ptm == "acetylation":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Acetylation (K)/"
        elif ptm == "Succinylation_K" or ptm == "succinyllysine":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Succinylation (K)/"
        elif ptm == "NlinkedGlycosylation_N" or ptm == "n_glycosylation":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/N-linked Glycosylation (N)/"
        elif ptm == "OlinkedGlycosylation_S":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/O-linked Glycosylation (S,T)/"
        elif ptm == "OlinkedGlycosylation_T" or ptm == "o_glycosylation":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/O-linked Glycosylation (S,T)/"
        elif ptm == "Methylation_R" or ptm == "methylation":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Methylation (R)/"
        elif ptm == "S-nitrosocysteine_C":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/S-nitrosylation (C)/"
        elif ptm == "Palmitoylation_C":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/S-palmitoylation (C)/"
        elif ptm == "Hydroxylation_K":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Hydroxylation (K)/"
        elif ptm == "Hydroxylation_P":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Hydroxylation (P)/"
        elif ptm == "Amidation_V":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Amidation (V)/"
        elif ptm == "Methylation_K":
            folder = "/mnt/pixstor/xudong-lab/yehan/PTMGPT2/new_model/Methylation (K)/"

        file = folder
        df, f1,acc, mcc, prc, rec, avg = get_score(file, test_data_loader)
        print(file)
        print("f1,acc,mcc,precision,recall,avg:", f1,acc, mcc, prc, rec, avg)
        with open(input_folder + 'test_results-Prompt1.csv', 'a') as f:
            f.write(f'{ptm},{f1},{acc},{mcc},{prc},{rec},{avg}\n')

        result_df = pd.DataFrame({
            "label": df["label"],
            "prediction": df["prediction"],
            "prot_id": data["UniprotID"],
            "position":data["Position"]
        })

        if balanced == True:
            result_df.to_csv(file + "/test_balanced_result.csv", index=False)
        else:
            result_df.to_csv(output_folder + ptm+"_test_output.csv", index=False)

        with open(os.path.join("/mnt/pixstor/data/yhhdb/PTMGPT/ptmgpt2_result/4_test_similarity/", "s2_10",ptm + "_test_similarity.txt"),
                  'r') as file:
            test_score_dic = json.load(file)

        thesold_list = [x * 10 for x in range(4, 12)]
        acc_list = []
        # roc_auc_list = []
        # pr_auc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        mcc_list = []

        for threshold in thesold_list:
            prediction_items = []
            label_items = []
            # print(f"The result for sequence with similiarity less than {threshold}:")
            for _, row in result_df.iterrows():
                # Protein ID
                protein_id = row["prot_id"]  # Replace with your UniProt Protein ID
                score = float(test_score_dic[protein_id])
                if score < threshold:
                    prediction_items.append(row["prediction"])
                    label_items.append(row["label"])

            # roc_auc = roc_auc_score(label_items, prediction_items)
            # # Compute precision-recall pairs
            # precision, recall, _ = precision_recall_curve(label_items, prediction_items)
            # pr_auc=average_precision_score(label_items, prediction_items)

            # Calculate PR AUC
            # pr_auc = auc(recall, precision)

            # Define a threshold
            # t = 0.5

            # Convert y_score to binary predictions
            # y_pred = (np.array(prediction_items) >= t).astype(int)
            y_pred=prediction_items
            precision = precision_score(label_items, y_pred)
            recall = recall_score(label_items, y_pred)
            f1 = f1_score(label_items, y_pred)
            mcc = matthews_corrcoef(label_items, y_pred)
            acc = accuracy_score(label_items, y_pred)

            acc_list.append(round(acc, 3))
            # roc_auc_list.append(round(roc_auc, 3))
            # pr_auc_list.append(round(pr_auc, 3))
            precision_list.append(round(precision, 3))
            recall_list.append(round(recall, 3))
            f1_list.append(round(f1, 3))
            mcc_list.append(round(mcc, 3))

            metrics_result = {"threshold": thesold_list,
                              "acc": acc_list,
                              "f1": f1_list,
                              "mcc": mcc_list,
                              # "roc_auc": roc_auc_list,
                              # "pr_auc": pr_auc_list,
                              "precision": precision_list,
                              "recall": recall_list }
            print(
                f"Threshold:{threshold};Accuracy: {acc:.3f}; F1: {f1:.3f}; Mcc: {mcc:.3f};Precision: {precision:.3f}; Recall: {recall:.3f} ")

        df_metric = pd.DataFrame(metrics_result)

        df_metric.to_csv(os.path.join("/mnt/pixstor/data/yhhdb/PTMGPT/ptmgpt2_result/5_test_metric_results/", ptm + '_metric_output.csv'),
                         index=False)

    # # results = []
    # folder="/mnt/pixstor/xudong-lab/yehan/results-Prompt1/n_glycosylation/"
    # # Replace the path with the output directory used during model training
    # for mdl in os.listdir(folder):
    #     if 'checkpoint' in mdl:
    #         mdl_path =folder + mdl
    #         print(mdl_path)
    #         f1, mcc, prc, rec, avg = get_score(mdl_path,test_data_loader)
    #         print("f1, mcc, prc, rec, avg:",f1, mcc, prc, rec, avg)
    #         with open(folder+'val_results-Prompt1.csv', 'a') as f:
    #             f.write(f'{mdl},{f1},{mcc},{prc},{rec},{avg}\n')





