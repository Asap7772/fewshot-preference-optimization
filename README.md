# FSPO: Few-Shot Preference Optimization

Code for [FSPO: Few-Shot Preference Optimization of Synthetic Data Elicits LLM Personalization to Real Users](https://www.arxiv.org/abs/2502.19312). 


## What is in this repo?
This repo provides code to train personalized models using FSPO. It is built on top of [Eric Mitchell's DPO codebase](https://github.com/eric-mitchell/direct-preference-optimization). The core modifications are in the `preference_datasets.py` file which contains dataloaders for FSPO. Additionally, due to the increased lengths of prompts with FSPO, we utilize [Flash Attention](https://github.com/Dao-AILab/flash-attention) to speed up training. 

## Setup
Create a venv with requirements listed in requirements.txt, ideally with python 3.12.
```bash
conda create --name FSPO python=3.12
source activate FSPO
pip install -r requirements.txt
```
Additionally, set the HF_TOKEN and WANDB_API_KEY environment variables.

## Running
To train a model, use the directpreferenceoptimization codebase. We added a dataloader so you can pass in the path to the preference dataset as a dataset. 
```bash
python -u train.py model=llama3-2-3b datasets=[roleplay] n_epochs=1 loss=sft lr=1e-7 exp_name=roleplay_prefft trainer=FSDPTrainer sample_during_eval=false eval_every=10000  do_first_eval=false debug=false wandb.project=personalization batch_size=4 max_prompt_length=8192 max_length=8192 eval_batch_size=4

python -u train.py model=llama3-2-3b datasets=[roleplay] n_epochs=1 loss=ipo lr=1e-6 loss.beta=0.01 exp_name=roleplay_ipo trainer=FSDPTrainer sample_during_eval=false eval_every=10000  do_first_eval=false debug=false wandb.project=personalization batch_size=4 max_prompt_length=8192 max_length=8192 eval_batch_size=4 model.archive=/PATH_TO_SFT_OUTPUT/LATEST/policy.pt
```
## Data
Along with this codebase, we also release the following datasets on HuggingFace:
- [Roleplay](https://huggingface.co/datasets/sher222/persona-iterative-responses)
- [Review](https://huggingface.co/datasets/Asap7772/steered_reviews_full_autolabel_gpt4o_pref)
- [Elix](https://huggingface.co/datasets/Asap7772/elix_generations_gpt4omini_pref)

## BibTeX
```
@misc{singh2025fspofewshotpreferenceoptimization,
      title={FSPO: Few-Shot Preference Optimization of Synthetic Preference Data in LLMs Elicits Effective Personalization to Real Users}, 
      author={Anikait Singh and Sheryl Hsu and Kyle Hsu and Eric Mitchell and Stefano Ermon and Tatsunori Hashimoto and Archit Sharma and Chelsea Finn},
      year={2025},
      eprint={2502.19312},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.19312}, 
}
```