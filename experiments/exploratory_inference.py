import sys
sys.path.append('../')
from utils import load_quantized_model_and_tokenizer, assess_device_memory, generate_text
from data.preprocess_data import load_data_from_csv

models = [
"meta-llama/Llama-2-7b-chat-hf",
"meta-llama/Llama-2-7b-hf",
"meta-llama/Llama-2-13b-chat-hf",
"meta-llama/Llama-2-13b-hf",
"meta-llama/Llama-2-70b-chat-hf",
"meta-llama/Llama-2-70b-hf",
]

if __name__=="__main__":

    #Load data
    data = load_data_from_csv("../data/inference.csv")
    results = data.copy()
    model_name = models[0]
    results[model_name] = "" #make empty column for results from this model

    #Load model, tokenizer
    model, tokenizer = load_quantized_model_and_tokenizer(model_name)
    assess_device_memory()

    #iterate over models TODO
    model_name = models[0]
    results[model_name] = "" #make empty column for results from this model

    #add column to csv for model name, append generated text to that column
    for index, row in data.iterrows():
        prompt = row['prompt']
        #better understand what generate text produces TODO
        text = generate_text(text="Tell me a sad story.", model=model, tokenizer=tokenizer)
        results.loc[index, model_name] = text
    
    #Save results TODO
    
    print(results.head())
    #generate_text(text="Tell me a sad story.", model=model, tokenizer=tokenizer)