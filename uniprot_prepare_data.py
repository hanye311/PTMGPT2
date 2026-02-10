import os
import re
from tqdm import tqdm
import pandas as pd
import subprocess
from Bio import SeqIO
import numpy as np
from sklearn.model_selection import KFold,train_test_split
import itertools
def collect_sequence_into_fasta(ptm_name, mahdi_ptm_read_csv_folder,mahdi_ptm_write_fasta_folder):
    ptm_count={}

    fasta_string=""
    ptm_df=pd.read_csv(mahdi_ptm_read_csv_folder+ptm_name+".csv")
    print("extract sequence from:"+ptm_name)
    for i in tqdm(range(ptm_df.shape[0])):
        id=ptm_df.at[i,'_id']
        sequence=ptm_df.at[i,'sequence']

        species = ptm_df.at[i, "species"]
        species = re.sub('[\[\]\'\'.]', "", species)
        species= species.split(",")[-1]

        date_time = ptm_df.at[i, 'data_time']
        # evidence_code_string = ptm_df.iat[i, 7]

        fasta_string=">"+id+"[organism="+species+"]"+"[date_time=" + date_time + "]"+"\n"+sequence+"\n"
        with open(mahdi_ptm_write_fasta_folder+ptm_name+'_sequence.fasta','a') as writers:
            writers.write(fasta_string)
    writers.close()
    print(ptm_name+" fasta saved!")

    return ptm_df.shape[0]


def run_cd_hit(input_file, output_file, identity_threshold, word_length,tolerance_for_redundance):
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
            '-n', str(word_length),
            '-t', str(tolerance_for_redundance)

        ]

        # Run the command
        subprocess.run(command, check=True)

        print(f"CD-HIT completed successfully. Output saved to {output_file}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running CD-HIT: {e}")

def get_sepecis_name(x):
    lineage=x.replace(" ", "").replace("[", "").replace("]", "").replace("'", "").split(",")
    if len(lineage)>=2:
        return lineage[1]
    else:
        return lineage[0]

def read_txt_to_list(file_name):
    list1 = []

    with open(file_name, 'r') as fh:
        for line in fh:
            list1.append(line.strip())
    return list1

def get_split_result(ptm_name,split_folder,csv_folder,dataset_folder):
    test_list = read_txt_to_list(split_folder + ptm_name + "/test.txt")
    unitprot_df = pd.read_csv(csv_folder + ptm_name + ".csv")
    criterion3 =unitprot_df['_id'].map(lambda x: x in test_list)
    test_df = unitprot_df[criterion3]

    test_df = test_df.rename(
        columns={
            "_id": "Uniprotid",
            "ac":"Uniprotac",
            "sequence": "Sequence",
            "data_time":"Time",
            "amino acid":"aminoacid",
            "species":"lineage"
        }
    )
    test_df["Length"] = test_df.apply(lambda row: len(row.Sequence),axis=1)
    test_df["position"]=test_df.apply(lambda row:row.position.replace("'", ""),axis=1)
    if not os.path.exists(dataset_folder+'test/'):
        os.makedirs(dataset_folder+'test/')
    test_df.to_csv(dataset_folder+'test/' + "final_"+ptm_name+'.csv', index=False)

    fold=5
    for i in range(fold):
        train_list=read_txt_to_list(split_folder+ptm_name+"/fold_"+str(i)+"/train.txt")
        val_list = read_txt_to_list(split_folder +ptm_name+"/fold_"+str(i)+ "/val.txt")

        criterion1 = unitprot_df['_id'].map(lambda x: x in train_list)
        train_df = unitprot_df[criterion1]

        criterion2 = unitprot_df['_id'].map(lambda x: x in val_list)
        val_df = unitprot_df[criterion2]

        train_df = train_df.rename(
            columns={
                "_id": "Uniprotid",
                "ac": "Uniprotac",
                "sequence": "Sequence",
                "data_time": "Time",
                "amino acid": "aminoacid",
                "species": "lineage"
            }
        )
        train_df["Length"] = train_df.apply(lambda row: len(row.Sequence), axis=1)
        train_df["position"] = train_df.apply(lambda row: row.position.replace("'", ""), axis=1)

        val_df = val_df.rename(
            columns={
                "_id": "Uniprotid",
                "ac": "Uniprotac",
                "sequence": "Sequence",
                "data_time": "Time",
                "amino acid": "aminoacid",
                "species": "lineage"
            }
        )
        val_df["Length"] = val_df.apply(lambda row: len(row.Sequence), axis=1)
        val_df["position"] = val_df.apply(lambda row: row.position.replace("'", ""), axis=1)

        if not os.path.exists(dataset_folder+'train/fold_'+str(i)):
            os.makedirs(dataset_folder+'train/fold_'+str(i))

        if not os.path.exists(dataset_folder+'valid/fold_'+str(i)):
            os.makedirs(dataset_folder+'valid/fold_'+str(i))

        train_df.to_csv(dataset_folder+'train/fold_'+str(i) + "/final_"+ptm_name+'.csv', index=False)
        val_df.to_csv(dataset_folder+'valid/fold_' +str(i) + "/final_"+ptm_name+'.csv', index=False)


    print(ptm_name + " datasets are saved.")

def get_fasta_of_each_csv():
    uniprot_ptm_read_csv_folder="uniprot/original_csv/"
    uniprot_ptm_write_fasta_folder='uniprot/fasta_all/'

    for csv_file in os.listdir(uniprot_ptm_read_csv_folder):
        if csv_file.split(".")[-1]=='csv':
            ptm_name=csv_file[:-4]
            if ptm_name+"_sequence.fasta" not in os.listdir(uniprot_ptm_write_fasta_folder):
                ptm_count = collect_sequence_into_fasta(ptm_name, uniprot_ptm_read_csv_folder, uniprot_ptm_write_fasta_folder)

def do_cd_hit_c():
    uniprot_ptm_read_fasta_folder="uniprot/fasta_all/"
    uniprot_ptm_save_cd_hit_folder='uniprot/cd-hit_result/c=0.4/'

    for fasta_file in os.listdir(uniprot_ptm_read_fasta_folder):
        if fasta_file.split(".")[-1]=='fasta':
            ptm_name = fasta_file[:-15]
            if ptm_name+".clstr" not in os.listdir(uniprot_ptm_save_cd_hit_folder):
                input_fasta=uniprot_ptm_read_fasta_folder + fasta_file
                output_clustered = uniprot_ptm_save_cd_hit_folder +ptm_name
                c=0.4
                n=2
                t=1
                run_cd_hit(input_fasta, output_clustered,c,n,t)


def generate_new_csv_according_to_cd_hit_result():
    uniprot_ptm_read_cd_hit_result_folder="uniprot/cd-hit_result/c=0.4/"
    uniprot_ptm_original_csv_folder = 'uniprot/original_csv/'
    uniprot_ptm_save_csv_folder='uniprot/csv_after_remove_rebundence/'

    for cd_hit_result_file in os.listdir(uniprot_ptm_read_cd_hit_result_folder):
        if cd_hit_result_file.split(".")[-1]!='clstr' and cd_hit_result_file.split(".")[-1]!='DS_Store':
            if cd_hit_result_file+".csv" not in os.listdir(uniprot_ptm_save_csv_folder):
                fasta_file=uniprot_ptm_read_cd_hit_result_folder+cd_hit_result_file
                orininal_csv_file = uniprot_ptm_original_csv_folder + cd_hit_result_file + ".csv"
                df_csv=pd.read_csv(orininal_csv_file)
                uniprotid_list=[]
                for seq_record in SeqIO.parse(fasta_file, "fasta"):
                    uniprotid = seq_record.id.split("[")[0]
                    uniprotid_list.append(uniprotid)

                criterion1 = df_csv['_id'].map(lambda x: x in uniprotid_list)
                df_new_csv = df_csv[criterion1]
                new_csv_file = uniprot_ptm_save_csv_folder + cd_hit_result_file + ".csv"
                df_new_csv.to_csv(new_csv_file, index=False)
                print(cd_hit_result_file + " has been generated!")

def seperate_csv_according_to_species_of_lineage():
    csv_read_folder='uniprot/csv_after_remove_rebundence/'
    csv_save_folder='uniprot/csv_after_remove_rebundence_different_species/'
    species_name_list=['Metazoa','Viridiplantae']

    for file in os.listdir(csv_read_folder):
        if file.split('.')[-1]=='csv':
            ptm_csv_df=pd.read_csv(os.path.join(csv_read_folder,file))
        ptm_type=file[:-4]
        print(ptm_type)
        for species_name in species_name_list:
            criterion1=ptm_csv_df['species'].map(lambda x:get_sepecis_name(x)==species_name)
            df=ptm_csv_df[criterion1].reset_index(drop=True)
            if df.shape[0]!=0:
                df.to_csv(csv_save_folder+ptm_type+"_"+species_name+".csv",index=False)

def seperate_cd_hit_result_into_training_validation_and_testing_csv():
    read_all_remove_rebundence_csv= 'uniprot/csv_after_remove_rebundence_different_species/'

    save_dataset_result='uniprot/dataset_fold_splitting/'
    ptm_type = "Phosphotyrosine_Metazoa"
    # ptm_type = "Phosphoserine"

    csv_df = pd.read_csv(read_all_remove_rebundence_csv+ptm_type+".csv")
    X=np.array(csv_df["_id"].to_list())

    # Step 1: Split the dataset into train+validation set (80%) and test set (20%)
    X_train_val, X_test, _,_ = train_test_split(X, X, test_size=0.2, random_state=42)
    if not os.path.exists(save_dataset_result+ptm_type):
        os.makedirs(save_dataset_result+ptm_type)

    f = open(save_dataset_result + ptm_type + '/test.txt', "w")
    for i in list(X_test):
        f.write(i)
        f.write('\n')
    f.close()

    # Initialize KFold with 5 splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform the 5-fold cross-validation
    fold = 0
    for train_index, val_index in kf.split(X_train_val):
        print(f"Fold {fold}:")

        if not os.path.exists(save_dataset_result + ptm_type+"/fold_"+str(fold)):
            os.makedirs(save_dataset_result + ptm_type+"/fold_"+str(fold))

        # Split the data into train and validation sets for this fold
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]

        f = open(save_dataset_result + ptm_type+"/fold_"+str(fold) + '/train.txt', "w")
        for i in list(X_train):
            f.write(i)
            f.write('\n')
        f.close()

        f = open(save_dataset_result + ptm_type + "/fold_" + str(fold) + '/val.txt', "w")
        for i in list(X_val):
            f.write(i)
            f.write('\n')
        f.close()

        fold += 1

def get_split_result_according_to_result_5():
    ptm_type="Phosphotyrosine_Metazoa"

    read_dataset_name_result='uniprot/dataset_fold_splitting/'
    read_csv_folder='uniprot/csv_after_remove_rebundence_different_species/'
    write_dataset_csv_folder='uniprot/dataset_csv/'
    get_split_result(ptm_type,read_dataset_name_result,read_csv_folder,write_dataset_csv_folder)

def combine_different_df(combined_df,ptm_df):
    new_df = pd.DataFrame(columns=(
        '_id', 'ac', 'species', 'sequence', 'data_time', 'position', 'amino acid', 'length', 'evidence_code'))

    combined_id_list = combined_df['_id'].tolist()
    ptm_id_list = ptm_df['_id'].tolist()

    intersection_list = list(set(combined_id_list).intersection(set(ptm_id_list)))
    for intersection_sample in tqdm(intersection_list):

        combined_ptm_df_temp = combined_df[combined_df['_id'] == intersection_sample].reset_index()
        ptm_df_temp = ptm_df[ptm_df['_id'] == intersection_sample].reset_index()

        combined_position =str(combined_ptm_df_temp.at[0, 'position']).replace(" ", "")
        combined_position = re.sub('[\[\]\'\'.]', "", combined_position)
        combined_position_list = combined_position.split(",")

        ptm_position = str(ptm_df_temp.at[0, 'position']).replace(" ", "")
        ptm_position = re.sub('[\[\]\'\'.]', "", ptm_position)
        ptm_position_list = ptm_position.split(",")

        combined_amino_acid =str(combined_ptm_df_temp.at[0, 'amino acid']).replace(" ", "")
        combined_amino_acid = re.sub('[\[\]\'\'.]', "", combined_amino_acid)
        combined_amino_acid_list = combined_amino_acid.split(",")

        ptm_amino_acid = str(ptm_df_temp.at[0, 'amino acid']).replace(" ", "")
        ptm_amino_acid = re.sub('[\[\]\'\'.]', "", ptm_amino_acid)
        ptm_amino_acid_list = ptm_amino_acid.split(",")

        combined_evidence_code =str(combined_ptm_df_temp.at[0, 'evidence_code']).replace(" ", "")
        combined_evidence_code = re.sub('[\[\]\'\'.]', "", combined_evidence_code)
        combined_evidence_code_list = combined_evidence_code.split(",")

        ptm_evidence_code = str(ptm_df_temp.at[0, 'evidence_code']).replace(" ", "")
        ptm_evidence_code = re.sub('[\[\]\'\'.]', "", ptm_evidence_code)
        ptm_evidence_code_list = ptm_evidence_code.split(",")

        # '_id', 'ac', 'species', 'sequence', 'data_time', 'position', 'amino acid', 'length', 'evidence_code'


        new_id = intersection_sample
        new_ac = combined_ptm_df_temp.at[0, 'ac']
        new_species = combined_ptm_df_temp.at[0, 'species']

        assert combined_ptm_df_temp.at[0,'sequence']==ptm_df_temp.at[0,'sequence'], "sequences should be equal!"

        new_sequence = combined_ptm_df_temp.at[0, 'sequence']
        new_data_time = combined_ptm_df_temp.at[0, 'data_time']
        new_position = combined_position_list + ptm_position_list
        new_amino_acid = combined_amino_acid_list + ptm_amino_acid_list
        new_length = len(new_sequence)
        new_evidence_code=combined_evidence_code_list+ptm_evidence_code_list

        new_row = pd.Series(
            [new_id, new_ac, new_species, new_sequence, new_data_time, new_position, new_amino_acid,
             new_length, new_evidence_code], index=new_df.columns)

        new_df = new_df._append(new_row.to_frame().T)


    combined_left_list = list(filter(lambda fruit: fruit not in intersection_list, combined_id_list))
    ptm_left_list= list(filter(lambda fruit:fruit not in intersection_list,ptm_id_list))


    criterion1 = combined_df['_id'].map(lambda x: x in combined_left_list)
    criterion2 = ptm_df['_id'].map(lambda x: x in ptm_left_list)

    combined_df['length']=combined_df['sequence'].apply(len)
    ptm_df['length'] = ptm_df['sequence'].apply(len)


    new_combined_df=pd.concat([new_df,combined_df[criterion1],ptm_df[criterion2]]).reset_index(drop=True)


    # Function to check for 'S', 'T', 'Y' and concatenate matches
    def check_for_characters(row):
        matches = []
        if 'S' in row:
            matches.append('S')
        if 'T' in row:
            matches.append('T')
        if 'Y' in row:
            matches.append('Y')
        return ''.join(matches)

    new_combined_df['types']=new_combined_df['amino acid'].apply(check_for_characters)
    new_combined_df['number of types'] = new_combined_df['types'].apply(len)

    return new_combined_df


def combine_different_ptm_type_together():
    csv_folder="uniprot/csv_after_remove_rebundence_different_species/"
    combined_csv_save_folder="uniprot/combined_csv/"
    combined_ptm_name_list=["Phosphoserine_Metazoa","Phosphothreonine_Metazoa","Phosphotyrosine_Metazoa"]
    i=1
    combined_df= pd.read_csv(csv_folder + combined_ptm_name_list[0] +".csv")
    while i<len(combined_ptm_name_list):
        ptm_df = pd.read_csv(csv_folder + combined_ptm_name_list[i] +".csv")
        combined_df=combine_different_df(combined_df,ptm_df)
        i+=1

    combined_df.to_csv(combined_csv_save_folder + str(combined_ptm_name_list)+ '.csv', index=False)

def split_combined_csv_into_dataset():
    combined_csv_read_folder = "uniprot/combined_csv/"
    combined_dataset_save_folder='uniprot/combined_dataset/'
    combined_csv_file_name="['Phosphoserine_Metazoa', 'Phosphothreonine_Metazoa', 'Phosphotyrosine_Metazoa'].csv"
    combined_df = pd.read_csv(combined_csv_read_folder + combined_csv_file_name)

    amino_acids=['S','T','Y']
    # Generate all combinations of 2 and 3 items
    combinations_of_two = list(itertools.combinations(amino_acids, 2))
    combinations_of_three = list(itertools.combinations(amino_acids, 3))

    # Combine both lists
    combinations= combinations_of_two + combinations_of_three
    all_combinations=amino_acids+ [''.join(combination) for combination in combinations]
    print(all_combinations)

    df_train_all = pd.DataFrame()
    df_validation_all = pd.DataFrame()
    df_test_all = pd.DataFrame()

    for amino_acids in all_combinations:
        df=combined_df.loc[combined_df['types']==amino_acids]
        df=df.sample(frac=1).reset_index(drop=True)

        df = df.rename(
            columns={
                "_id": "Uniprotid",
                "ac": "Uniprotac",
                "sequence": "Sequence",
                "data_time": "Time",
                "amino acid": "aminoacid",
                "species": "lineage",
                "length":"Length"
            }
        )

        train_number=int(len(df)*0.8)
        validation_number = int(len(df)*0.1)
        test_number= int(len(df)*0.1)

        df_train=df[0:train_number]
        df_validation=df[train_number:train_number+validation_number]
        df_test=df[train_number+validation_number:]

        df_train_all = pd.concat([df_train_all,df_train]).reset_index(drop=True)
        df_validation_all = pd.concat([df_validation_all,df_validation]).reset_index(drop=True)
        df_test_all = pd.concat([df_test_all, df_test]).reset_index(drop=True)


    if not os.path.exists(combined_dataset_save_folder+"train/"):
        os.makedirs(combined_dataset_save_folder+"train/")

    if not os.path.exists(combined_dataset_save_folder+"valid/"):
        os.makedirs(combined_dataset_save_folder+"valid/")

    if not os.path.exists(combined_dataset_save_folder+"test/"):
        os.makedirs(combined_dataset_save_folder+"test/")

    df_train_all.to_csv(combined_dataset_save_folder+"train/final_Phosphorylation_Metazoa.csv",index=False)
    df_validation_all.to_csv(combined_dataset_save_folder+"valid/final_Phosphorylation_Metazoa.csv",index=False)
    df_test_all.to_csv(combined_dataset_save_folder + "test/final_Phosphorylation_Metazoa.csv",index=False)

if __name__ == "__main__":
    ## 1_get fasta of each csv
    get_fasta_of_each_csv()

    # # 2_do cd-hit -c=0.4   command:  cd-hit -i all_sequence.fasta -o /mnt/pixstor/data/yhhdb/cd-hit/0.4/all_sequence -c 0.4 -n 2 -T 8
    # do_cd_hit_c()

    # #3_generate new csv according to cd-hit result. get the designated one sequence from each cluster
    # generate_new_csv_according_to_cd_hit_result()

    ##4_seperate csv according to species of lineage
    # seperate_csv_according_to_species_of_lineage()

    ##5 seperate cd-hit result into training validation and testing csv
    # seperate_cd_hit_result_into_training_validation_and_testing_csv()

    # 6_get_split_result according to result 5
    # get_split_result_according_to_result_5()

    #7_combine different ptm type together
    # combine_different_ptm_type_together()

    #8_split_combined_csv into training,validation and testing dataset
    # split_combined_csv_into_dataset()
