# llm_bias
Investigating if we can find circuits in LLMs that reinforce human-biases found in training data

## Installation

Requirements: 
`python >=3.7,<3.11`

```
git clone https://github.com/msakarvadia/llm_bias.git
cd llm_bias
conda create -p env python==3.10 (or conda create --prefix=env python=3.10)
conda activate env
pip install -r requirements.txt
```

Getting [interactive node](https://docs.nersc.gov/jobs/interactive/) on perlmutter

`salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=mxxxx`
