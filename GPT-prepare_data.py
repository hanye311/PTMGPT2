import os.path

import pandas as pd
from tqdm import tqdm
import pickle
import re
import subprocess
# Function to split sequences and PTM sites into chunks
def split_into_windows(row,balanced):
    sequence = row['Sequence']

    ptm_sites = eval(row['Position'])
    uniprotid= row['Uniprotid']
    # Extract the positions of modified residues from the 'Modified residue' column
    # ptm_sites = ptm_sites.replace(" ", "")
    # ptm_sites = re.sub('[\[\]\'\'.]', "", ptm_sites)
    # ptm_sites = ptm_sites.split(",")
    # ptm_sites = [int(i) for i in ptm_sites]

    amino_acid=row['Aminoacid']
    # amino_acid = re.sub('[\[\]\'\'.]', "", amino_acid)[0]

    windows_size = 21
    # # Calculate the number of chunks
    # num_windows = len(sequence.find(amino_acid))

    sequence_length=len(sequence)
    df=pd.DataFrame(columns=('Seq','Label','Position','UniprotID'))

    # # sequence = "AABBCCDDEKASDF"
    # sequence = "AAAAAAAAAAKASDFAAAAA"
    # # sequence = "KASDFAAAAAAAAAAAAAAA"
    # # sequence = "AAAAAAAAAAAAAAAAAAAAAAAAKASDFBBBBBBBBBBBBBBBBBBBBB"
    # sequence_length=len(sequence)

    for i in range(sequence_length):
        if sequence[i]==amino_acid:
            if i<=9 and i+10>=sequence_length:
                Seq = sequence[:]
            elif i>9 and i+10>=sequence_length:
                Seq = sequence[i - 10:]
            elif i<=9 and i+10<sequence_length:
                Seq = sequence[:i + 11]
            else:
                Seq = sequence[i-10:i+11]
            if i+1 in ptm_sites:
                Label = 1
            else:
                Label = 0

            new_row = pd.Series([Seq,Label,i+1,uniprotid], index=df.columns)
            df = df._append(new_row.to_frame().T)

    if balanced:
        df_positives=df[df['Label']==1]
        df_negatives=df[df['Label']==0]
        if df_negatives.shape[0]>=df_positives.shape[0]:
            df_new=pd.concat([df_positives, df_negatives.sample(n=df_positives.shape[0])])
        else:
            df_new = pd.concat([df_positives.sample(n=df_negatives.shape[0]), df_negatives])

        return df_new
    else:
        return df

def run_cd_hit(input_file, output_file, identity_threshold, word_length):
    """
    Run CD-HIT on the input file.

    Parameters:
    - input_file (str): Path to the input FASTA file.
    - output_file (str): Path to the output file.
    - identity_threshold (float): Sequence identity threshold (default is 0.9).
    - word_length (int): Word length (default is 5 for proteins).
    Choose of word size:-n 5 for thresholds 0.7 ~ 1.0
    -n 4 for thresholds 0.6 ~ 0.7
    -n 3 for thresholds 0.5 ~ 0.6
    -n 2 for thresholds 0.4 ~ 0.5
    """
    try:
        # Construct the CD-HIT command
        command = [
            'cd-hit',
            '-i', input_file,
            '-o', output_file,
            '-c', str(identity_threshold),
            '-n', str(word_length)
        ]

        # Run the command
        subprocess.run(command, check=True)

        print(f"CD-HIT completed successfully. Output saved to {output_file}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running CD-HIT: {e}")

def collect_sequence_into_fasta(ptm_df,save_file):

    fasta_string=""

    for i in tqdm(range(ptm_df.shape[0])):
        sequence=ptm_df.iat[i,0]
        label=ptm_df.iat[i,1]

        fasta_string=fasta_string+">"+str(label)+"\n"+sequence+"\n"

    with open(save_file,'w') as writers:
        writers.write(fasta_string.strip())
    writers.close()
    print(save_file +" saved!")

def file_to_df(data_file,balanced):
    data = pd.read_csv(data_file)
    data=data[data["Length"]<=1024].reset_index()

    # Create a new DataFrame to store the chunks

    windows_df = pd.DataFrame(columns=('Seq','Label','Position','UniprotID'))

    # Iterate through each row of the original DataFrame and split into chunks
    for _, row in data.iterrows():

        df_return=split_into_windows(row,balanced)
        windows_df = pd.concat([windows_df, df_return])

    # # Convert the list of chunks into a DataFrame
    # windows_df = pd.DataFrame(windows_data)

    # Reset the index of the DataFrame
    windows_df.reset_index(drop=True, inplace=True)

    return windows_df
    # windows_df.to_csv(save_file)
def deal_with_cdhit_result(input_file,output_file):

    record_list=[]

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                record = {}
                record["Label"] = int(line[1:])
            else:
                record["Seq"]=str(line)
                record_list.append(record)

    df = pd.DataFrame(record_list)

    df.to_csv(output_file)

def read_txt_to_list(file_name):
    list1 = []

    with open(file_name, 'r') as fh:
        for line in fh:
            list1.append(line.strip())
    return list1

if __name__ == "__main__":
    # ptm_file="final_Ubiquitylation.csv"
    # #
    # # Load the TSV file
    # train_file_path = 'Mahdi_dataset/PTMs/train/'+ptm_file
    # train_df= file_to_df(train_file_path)
    #
    #
    # save_file='cd-hit_result/PTMGPT2/ubiquitylation/c=0.9/train_21nt.fasta'
    # collect_sequence_into_fasta(train_df,save_file)
    #
    # # # Example usage
    # c = 0.9
    # n = 2
    # input_fasta = 'cd-hit_result/PTMGPT2/ubiquitylation/c=0.9/train_21nt.fasta'
    # output_clustered = 'cd-hit_result/PTMGPT2/ubiquitylation/c=0.9/train_21nt'
    # run_cd_hit(input_fasta, output_clustered, c, n)
    #
    # input_file= "cd-hit_result/PTMGPT2/ubiquitylation/c=0.9/train_21nt"
    # output_file="GPT-dataset/ubiquitylation/train_remove_rebundancy_0.9.csv"
    # deal_with_cdhit_result(input_file,output_file)
    #
    # val_file_path='Mahdi_dataset/PTMs/valid/'+ptm_file
    # val_df = file_to_df(val_file_path)
    # val_df.to_csv("GPT-dataset/ubiquitylation/val.csv")
    #
    # test_file_path='Mahdi_dataset/PTMs/test/'+ptm_file
    # test_df = file_to_df(test_file_path)
    # test_df.to_csv("GPT-dataset/ubiquitylation/test.csv")
    #
    # print("train_count:",train_df['Label'].value_counts())
    # print("val_count:",val_df['Label'].value_counts())
    # print("test_count",test_df['Label'].value_counts())


    # ptm_list=['ubiquitylation','phosphorylation','acetylation','n_glycosylation','o_glycosylation','methylation']
    # ptm_name_file="GPT-dataset/ptm_20/ptm_20.txt"
    # ptm_list=read_txt_to_list(ptm_name_file)
    # input_folder="uniprot_dataset/"
    input_folder=os.path.join("ptmgpt2_result","1_test_protein_sequences")
    output_folder=os.path.join("ptmgpt2_result","2_test_21nt_peptides")
    # ptm_list=["Phosphoserine_Metazoa","Phosphotyrosine_Metazoa"]
    # ptm_list=[
    #            "Phosphothreonine_Metazoa",
    #            "Asymmetric dimethylarginine_Metazoa",
    #           "Glycyl lysine isopeptide (Lys-Gly) (interchain with G-Cter in SUMO)_Metazoa",
    #           "Glycyl lysine isopeptide (Lys-Gly) (interchain with G-Cter in SUMO1)_Metazoa",
    #           "Glycyl lysine isopeptide (Lys-Gly) (interchain with G-Cter in SUMO2)_Metazoa",
    #           "Glycyl lysine isopeptide (Lys-Gly) (interchain with G-Cter in ubiquitin)_Metazoa",
    #           "N6-acetyllysine_Metazoa",
    #           "N6-succinyllysine_Metazoa",
    #           "N-acetylalanine_Metazoa",
    #           "N-acetylmethionine_Metazoa",
    #           "N-acetylserine_Metazoa",
    #           "N-linked (GlcNAc...) (complex) asparagine_Metazoa",
    #           "N-linked (GlcNAc...) asparagine_Metazoa",
    #           "N-myristoyl glycine_Metazoa",
    #           "O-linked (GalNAc...) threonine_Metazoa",
    #           "Omega-N-methylarginine_Metazoa",
    #           "Pyrrolidone carboxylic acid_Metazoa",
    #           "S-palmitoyl cysteine_Metazoa",
    #           "N6-methyllysine_Metazoa",
    #           "N6,N6-dimethyllysine_Metazoa",
    #           "N6,N6,N6-trimethyllysine_Metazoa"]
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
    balanced=False
    for ptm in ptm_list:

        test_file_path=os.path.join(input_folder,"final_"+ptm+".csv")

        test_df = file_to_df(test_file_path,balanced)

        # if not os.path.exists('GPT-dataset/different_species/' + ptm):
        #     os.makedirs('GPT-dataset/different_species/' + ptm)

        if balanced:
            save_file=input_folder+ptm +'/test_balanced.csv'
            if test_df[test_df['Label']==0].shape[0] == test_df[test_df['Label']==1].shape[0]:
                test_df.to_csv(save_file)
            else:
                print("The return df is not a balanced dataset!")
        else:
            save_file = os.path.join(output_folder,ptm + '_21nt.csv')
            test_df.to_csv(save_file,index=False)





