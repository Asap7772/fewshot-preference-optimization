# FSPO: Few-Shot Preference Optimization

Code for [FSPO: Few-Shot Preference Optimization of Synthetic Data Elicits LLM Personalization to Real Users](https://arxiv.org/abs/). 


## What is in this repo?
This repo provides code to train personalized models using FSPO. It is built ontop of [Eric Mitchell's DPO codebase](https://github.com/eric-mitchell/direct-preference-optimization). The core modifications are in the `preference_datasets.py` file which contains dataloaders for FSPO. Additionally, due to the increased lengths of prompts with FSPO, we utilize [Flash Attention](https://github.com/Dao-AILab/flash-attention) to speed up training. 

## Setup
Create a venv with requirements listed in requirements.txt, ideally with python 3.12.
```bash
conda create --name FSPO python=3.12
source activate FSPO
pip install -r requirements.txt
```
Additionally, set the HF_TOKEN and WANDB_API_KEY environment variables.

## Running
To train a model, use the direct-preference-optimization codebase. We added a dataloader so you can pass in the path to the preference dataset as a dataset. 
```bash
python -u direct-preference-optimization/train.py model=llama3-8b datasets=[PATH_TO_SAMPLED_DATASET.json] n_epochs=1 loss=sft lr=1e-7 exp_name=gemma9b_sft trainer=FSDPTrainer sample_during_eval=false eval_every=1_000_000  do_first_eval=false debug=false wandb.project=rl-hotpotqa-finalize batch_size=8 max_prompt_length=2048 max_length=2048
python -u direct-preference-optimization/train.py model=llama3-8b datasets=[PATH_TO_SAMPLED_DATASET.json] n_epochs=2 loss=ipo lr=1e-7 loss.beta=0.05 exp_name=gemma9b_ipo trainer=FSDPTrainer sample_during_eval=false eval_every=1_000_000  do_first_eval=false debug=false wandb.project=rl-hotpotqa-finalize batch_size=4 max_prompt_length=2048 max_length=2048 model.archive=/PATH_TO_SFT_OUTPUT/LATEST/policy.pt
```
## Data
Along with this codebase, we also release the following datasets on HuggingFace:
- [Roleplay](https://huggingface.co/datasets/sher222/persona-iterative-responses)

<!-- ## BibTeX
```
@misc{hsu2024groundingtryingllmsreinforcement,
      title={Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval}, 
      author={Sheryl Hsu and Omar Khattab and Chelsea Finn and Archit Sharma},
      year={2024}
}
``` -->