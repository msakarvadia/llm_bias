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

### Black

To maintain consistent formatting, we take advantage of `black` via pre-commit hooks.
There will need to be some user-side configuration. Namely, the following steps:

1. Install black via `pip install black` (included in `requirements.txt`).
2. Install `pre-commit` via `pip install pre-commit` (included in `requirements.txt`).
3. Run `pre-commit install` to setup the pre-commit hooks.

Once these steps are done, you just need to add files to be committed and pushed and the hook will reformat any Python file that does not meet Black's expectations and remove them from the commit. Just re-commit the changes and it'll be added to the commit before pushing.


Getting [interactive node](https://docs.nersc.gov/jobs/interactive/) on perlmutter

`salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=mxxxx`
