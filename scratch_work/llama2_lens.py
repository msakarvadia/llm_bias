import sys
sys.path.append('../')
from utils import load_quantized_model_and_tokenizer, assess_device_memory, generate_text
from data.preprocess_data import load_data_from_csv, save_data_as_csv
import os.path
import pandas as pd

models = [
"meta-llama/Llama-2-70b-chat-hf",
"meta-llama/Llama-2-70b-hf",
"meta-llama/Llama-2-7b-chat-hf",
"meta-llama/Llama-2-7b-hf",
"meta-llama/Llama-2-13b-chat-hf",
"meta-llama/Llama-2-13b-hf",
]

if __name__=="__main__":

    #Load data
    data = load_data_from_csv("../data/inference.csv")
    results_file_path = "initial_inference_results.csv"
    if os.path.isfile(results_file_path):
        results = load_data_from_csv(results_file_path)
    else:
        results = data.copy()

    #iterate over models
    for m in models:
        model_name = m

        #check if we have already started processing for a given model
        if not model_name in results.columns:
            results[model_name] = "" #make empty column for results from this model

        #Load model, tokenizer
        model, tokenizer = load_quantized_model_and_tokenizer(model_name)
        assess_device_memory()

        #Load model into TransformerLens template


        """
        #add column to csv for model name, append generated text to that column
        for index, row in data.iterrows():
            #skip rows that already have content
            if not pd.isna(results.loc[index, model_name]):
                continue
            prompt = row['prompt']
            #better understand what generate text produces TODO
            text = generate_text(text=prompt, model=model, tokenizer=tokenizer)
            results.loc[index, model_name] = text
            if index%50 == 0:
                print(results.tail())
                #Save results
                save_data_as_csv(df=results, filename=results_file_path)
        """
