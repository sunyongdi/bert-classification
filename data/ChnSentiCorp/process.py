import pandas as pd
import csv

labels = []

csv_file = '/root/sunyd/nlp_tutorial/text-classification/bert-classification/data/ChnSentiCorp/train.tsv' 


data = pd.read_csv(csv_file, sep='\t')

for row in data.iloc:
    labels.append(str(row['label']))
    
labels = set(labels)
print(labels)
with open('./labels.txt','w') as fp:
  fp.write("\n".join(labels))