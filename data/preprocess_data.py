"""This file preprocesses exploratory data."""

import pandas as pd

def load_data_from_csv(filename:str) -> pd.DataFrame:
    """ Load data from csv and return pandas df"""
    return pd.read_csv(filename)

def save_data_as_csv(df: pd.DataFrame,filename:str):
    """Save data from pandas df to csv"""
    return df.to_csv(filename, index=False)

def create_exploratory_inference_data(input_filename:str="bias_data.csv", output_filename:str="inference.csv"):
    data = load_data_from_csv(input_filename)

    categories= ['gender', 'education', 'race', 'geography', 'politics']
    
    inference = pd.DataFrame(columns = ['prompt', 'category'])

    for q in data['questions']:
        for c in categories:
            for pre in data[c].dropna():
                prompt = pre+" "+q
                row = {'prompt':prompt,
                       'category':c}
                inference.loc[len(inference)] = row
                #print(pre+" "+q)
    #print(inference.head())
    save_data_as_csv(df=inference, filename="inference.csv")

if __name__=="__main__":
    create_exploratory_inference_data("bias_data.csv")