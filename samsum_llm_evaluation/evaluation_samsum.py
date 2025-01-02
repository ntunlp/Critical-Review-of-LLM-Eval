import evaluate, glob, csv, pandas as pd
rouge = evaluate.load('rouge')
all_dataframes = []
path = 'samsum_response'
all_files = glob.glob(path + "/*.csv")
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    all_dataframes.append([filename,df])

gold_df = pd.read_csv('samsum_dataset.csv')
references = []
for index, row in gold_df.iterrows():
    references.append(str(row['reference']))

for data in all_dataframes:
    filename = data[0]
    df = data[1]
    predictions = []
    for index, row in df.iterrows():
        predictions.append(str(row['response']))
    print(filename)
    print(len(references),len(predictions))
    results = rouge.compute(predictions=predictions,references=references)
    print(results)
    print("####")
