import datasets
import torch
from utils import TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from transformers import AutoTokenizer


def shots_to_text(
    fewshot_prompts: List[str], 
    fewshot_preferred: List[str], 
    fewshot_dispreferred: List[str], 
    final_question: str, 
    final_chosen: str, 
    final_rejected: str, 
    persona_chosen: str = None, 
    persona_rejected: str = None, 
    include_dispreferred: bool = False, 
    add_persona: bool = False, 
    teacher_forced: bool = False
):
    """Convert a list of few-shot examples and a final question to a single string using markdown."""
    assert len(fewshot_prompts) == len(fewshot_preferred) == len(fewshot_dispreferred), "Few-shot examples must have the same length"
    num_shots = len(fewshot_prompts)
    fewshot_text = ""
    for i, (prompt, preferred, dispreferred) in enumerate(zip(fewshot_prompts, fewshot_preferred, fewshot_dispreferred)):
        fewshot_text += "# Example " + str(i + 1) + "\n"
        fewshot_text += "## Question \n" + prompt + "\n"
        if include_dispreferred:
            fewshot_text += "## Preferred Response \n" + preferred + "\n"
            fewshot_text += "## Dispreferred Response \n" + dispreferred + "\n"
        else:
            fewshot_text += "## Response \n" + preferred + "\n"
        fewshot_text += "\n"
    
    fewshot_text += "# Task\n"
    
    if num_shots == 0:
        task_str = "Generate a response to the following question."
    elif include_dispreferred and add_persona:
        task_str = "Generate a user description based on the examples above. Then, generate a preferred response to the following question."
    elif add_persona:
        task_str = "Generate a user description based on the examples above. Then, generate a response to the following question."
    elif include_dispreferred:
        task_str = "Given the examples above, generate a preferred response to the following question."
    else:
        task_str = "Given the examples above, generate a response to the following question."
    
    fewshot_text += task_str + "\n"
    fewshot_text += "## Question \n" + final_question + "\n"
    
    if include_dispreferred:
        final_chosen = "## Preferred Response \n" + final_chosen + "\n"
        final_rejected = "## Preferred Response \n" + final_rejected + "\n"
    else:
        final_chosen = "## Response \n" + final_chosen + "\n"
        final_rejected = "## Response \n" + final_rejected + "\n"

    chosen, rejected = final_chosen, final_rejected
    if add_persona:
        prefix_persona_chosen = "## User Description\n" + persona_chosen + "\n"
        prefix_persona_rejected = "## User Description\n" + persona_rejected + "\n"
        if teacher_forced:
            fewshot_text = fewshot_text + prefix_persona_chosen
        else:
            chosen = prefix_persona_chosen + final_chosen
            rejected = prefix_persona_rejected + final_rejected
    
    return fewshot_text, chosen, rejected

def get_elix(
    split: str, 
    silent: bool=False, 
    cache_dir: str=None, 
    num_shots=4, 
    include_disprefferred=True, 
    add_persona=False, 
    use_scorer_persona=True,
    teacher_forced=False, 
    include_level=True, 
    autolabel=False,
    multi_user=False,
    scorer_match=False,
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading ELIX (regen) dataset ({split} split) from Huggingface...')
    if autolabel:
        dataset = datasets.load_dataset('Asap7772/elix_generations_autolabel', split=split, cache_dir=cache_dir)
    elif multi_user:
        dataset = datasets.load_dataset('Asap7772/elix_multexpert_preferences_gpt4o_pref', split=split, cache_dir=cache_dir)
    else:
        dataset = datasets.load_dataset('Asap7772/elix_generations_gpt4omini_pref', split=split, cache_dir=cache_dir)
    df = dataset.to_pandas()
    print('done')
    
    if scorer_match:
        df = df[(df['scorer_level'] == df['level_x']) | (df['scorer_level'] == df['level_y'])]
    
    data = defaultdict(lambda: defaultdict(list))
    for _ in range(2 * (num_shots + 1)):
        for level, level_df in tqdm.tqdm(df.groupby('scorer_level'), desc='Processing levels', disable=silent):
            level_df = level_df.sample(frac=1).reset_index(drop=True)
            index_df = 0
            while index_df < len(level_df):
                start = index_df
                end = index_df + (num_shots + 2) # get N few-shot examples and 1 prompt
                if end > len(level_df):
                    break
                rows = level_df.iloc[start:end]
                
                fewshot_prompts = []
                fewshot_preferred = []
                fewshot_dispreferred = []
                for shot_idx in range(num_shots):
                    x = rows['prompt'].iloc[shot_idx].strip()
                    y1 = rows['response_x'].iloc[shot_idx].strip() 
                    y2 = rows['response_y'].iloc[shot_idx].strip()
                                        
                    label = rows['label'].iloc[shot_idx]
                    yw = y1 if label == 0 else y2
                    yl = y2 if label == 0 else y1
                    
                    fewshot_prompts.append(x)
                    fewshot_preferred.append(yw)
                    fewshot_dispreferred.append(yl)
                
                persona_scorer = f"The user is {level}."
                final_question = rows['prompt'].iloc[num_shots].strip()
                
                y1_last = rows['response_x'].iloc[num_shots].strip()
                y2_last = rows['response_y'].iloc[num_shots].strip()
                level_y1_last = rows['level_x'].iloc[num_shots]
                level_y2_last = rows['level_y'].iloc[num_shots]
                label_last = rows['label'].iloc[num_shots]
                final_chosen = y1_last if label_last == 0 else y2_last
                level_chosen = level_y1_last if label_last == 0 else level_y2_last
                persona_chosen = f"The user is {level_chosen}."
                final_rejected = y2_last if label_last == 0 else y1_last
                level_rejected = level_y2_last if label_last == 0 else level_y1_last
                persona_rejected = f"The user is {level_rejected}."
                
                prompt, chosen, rejected = shots_to_text(
                    fewshot_prompts=fewshot_prompts,
                    fewshot_preferred=fewshot_preferred,
                    fewshot_dispreferred=fewshot_dispreferred,
                    final_question=final_question,
                    final_chosen=final_chosen,
                    final_rejected=final_rejected,
                    persona_chosen=persona_scorer if use_scorer_persona else persona_chosen,
                    persona_rejected=persona_rejected,
                    include_dispreferred=include_disprefferred,
                    add_persona=add_persona,
                    teacher_forced=teacher_forced,
                )
                responses = [chosen, rejected]
                n_responses = len(data[prompt]['responses'])
                data[prompt]['pairs'].append((n_responses, n_responses + 1))
                data[prompt]['responses'].extend(responses)
                data[prompt]['sft_target'].extend([chosen, rejected])
                if include_level:
                    data[prompt]['level'].append(level)
                
                index_df = end
    return data

def get_review(
    split: str, 
    silent: bool=False, 
    cache_dir: str=None, 
    num_shots=4, 
    include_disprefferred=True, 
    add_persona=False, 
    use_scorer_persona=True,
    teacher_forced=False, 
    include_level=True, 
    autolabel=False,
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading Review dataset ({split} split) from Huggingface...')
    if autolabel:
        dataset = datasets.load_dataset('Asap7772/steered_reviews_full_autolabel', split=split, cache_dir=cache_dir)
    else:
        dataset = datasets.load_dataset('Asap7772/steered_reviews_full_autolabel_gpt4o_pref', split=split, cache_dir=cache_dir)
    df = dataset.to_pandas()
    print('done')
    
    data = defaultdict(lambda: defaultdict(list))
    for _ in range(2 * (num_shots + 1)):
        for level, level_df in tqdm.tqdm(df.groupby('scorer_level'), desc='Processing levels', disable=silent):
            level_df = level_df.sample(frac=1).reset_index(drop=True)
            index_df = 0
            while index_df < len(level_df):
                start = index_df
                end = index_df + (num_shots + 2) # get N few-shot examples and 1 prompt
                if end > len(level_df):
                    break
                rows = level_df.iloc[start:end]
                
                fewshot_prompts = []
                fewshot_preferred = []
                fewshot_dispreferred = []
                for shot_idx in range(num_shots):
                    x = rows['prompt'].iloc[shot_idx].strip()
                    y1 = rows['response_x'].iloc[shot_idx].strip() 
                    y2 = rows['response_y'].iloc[shot_idx].strip()
                                        
                    label = rows['label'].iloc[shot_idx]
                    yw = y1 if label == 0 else y2
                    yl = y2 if label == 0 else y1
                    
                    fewshot_prompts.append(x)
                    fewshot_preferred.append(yw)
                    fewshot_dispreferred.append(yl)
                
                mapped_level = {
                    'negative': 'This user prefers negative reviews.',
                    'positive': 'This user prefers positive reviews.',
                    'concise': 'This user prefers concise reviews.',
                    'verbose': 'This user prefers verbose reviews.',
                    'positive+concise': 'This user prefers positive and concise reviews.',
                    'negative+concise': 'This user prefers negative and concise reviews.',
                    'positive+verbose': 'This user prefers positive and verbose reviews.',
                    'negative+verbose': 'This user prefers negative and verbose reviews.',
                }
                persona_scorer = mapped_level[level]
                final_question = rows['prompt'].iloc[num_shots].strip()
                
                y1_last = rows['response_x'].iloc[num_shots].strip()
                y2_last = rows['response_y'].iloc[num_shots].strip()
                level_y1_last = rows['level_x'].iloc[num_shots]
                level_y2_last = rows['level_y'].iloc[num_shots]
                label_last = rows['label'].iloc[num_shots]
                final_chosen = y1_last if label_last == 0 else y2_last
                level_chosen = level_y1_last if label_last == 0 else level_y2_last
                persona_chosen = f"The user is {level_chosen}."
                final_rejected = y2_last if label_last == 0 else y1_last
                level_rejected = level_y2_last if label_last == 0 else level_y1_last
                persona_rejected = f"The user is {level_rejected}."
                
                prompt, chosen, rejected = shots_to_text(
                    fewshot_prompts=fewshot_prompts,
                    fewshot_preferred=fewshot_preferred,
                    fewshot_dispreferred=fewshot_dispreferred,
                    final_question=final_question,
                    final_chosen=final_chosen,
                    final_rejected=final_rejected,
                    persona_chosen=persona_scorer if use_scorer_persona else persona_chosen,
                    persona_rejected=persona_rejected,
                    include_dispreferred=include_disprefferred,
                    add_persona=add_persona,
                    teacher_forced=teacher_forced,
                )
                responses = [chosen, rejected]
                n_responses = len(data[prompt]['responses'])
                data[prompt]['pairs'].append((n_responses, n_responses + 1))
                data[prompt]['responses'].extend(responses)
                data[prompt]['sft_target'].extend([chosen, rejected])
                if include_level:
                    data[prompt]['level'].append(level)
                
                index_df = end
    return data


def get_roleplay(
    split: str, 
    silent: bool=False, 
    cache_dir: str=None, 
    num_shots=8, 
    include_disprefferred=True, 
    add_persona=False,
    include_level=False
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading roleplay dataset ({split} split) from Huggingface...')
    dataset_name = f"sher222/persona-iterative-responses"
    dataset = datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    print("loading data from", dataset_name)
    df = dataset.to_pandas()

    data = defaultdict(lambda: defaultdict(list))
    num_loops = 1 if split == "test" else 2 * (num_shots + 1)
    for _ in range(num_loops):
        for level, level_df in tqdm.tqdm(df.groupby('level'), desc='Processing levels', disable=silent):
            level_df = level_df.sample(frac=1).reset_index(drop=True)
            i = 0
            while i < len(level_df):
                start = i
                end = i + (num_shots + 2) # get N few-shot examples and 1 prompt
                if end > len(level_df):
                    break
                rows = level_df.iloc[start:end]
                
                # construct prompt
                fewshot_prompts = []
                fewshot_preferred = []
                fewshot_dispreferred = []
                for shot_idx in range(num_shots):
                    x = rows['x'].iloc[shot_idx].strip()
                    yw = rows['yw'].iloc[shot_idx].strip()
                    yl = rows['yl'].iloc[shot_idx].strip()
                    fewshot_prompts.append(x)
                    fewshot_preferred.append(yw)
                    fewshot_dispreferred.append(yl)

                response_row = rows.iloc[[num_shots]].to_dict('records')[0]
                chosen, rejected = response_row['yw'].strip(), response_row['yl'].strip()

                prompt, chosen, rejected = shots_to_text(
                    fewshot_prompts=fewshot_prompts,
                    fewshot_preferred=fewshot_preferred,
                    fewshot_dispreferred=fewshot_dispreferred,
                    final_question=response_row["x"].strip(),
                    final_chosen=response_row["yw"].strip(),
                    final_rejected=response_row["yl"].strip(),
                    include_dispreferred=include_disprefferred,
                    add_persona=add_persona,
                )

                responses = [chosen, rejected]
                n_responses = len(data[prompt]['responses'])
                data[prompt]['pairs'].append((n_responses, n_responses + 1))
                data[prompt]['responses'].extend(responses)
                data[prompt]['sft_target'].extend([chosen, rejected])
                if include_level:
                    data[prompt]['level'].append(response_row['score_persona'])

                i = end
    return data



def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, include_level=False):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'elix':
        data = get_elix(split, silent=silent, cache_dir=cache_dir, include_disprefferred=True, include_level=include_level, autolabel=False)
    elif name == 'review_4shot':
        data = get_review(split, silent=silent, cache_dir=cache_dir, include_disprefferred=True, include_level=include_level, autolabel=False, num_shots=8)
    elif name == 'review_8shot':
        data = get_review(split, silent=silent, cache_dir=cache_dir, include_disprefferred=True, include_level=include_level, autolabel=False, num_shots=8)
    elif name == 'roleplay':
        data = get_roleplay(split, silent=silent, cache_dir=cache_dir, num_shots=8, include_disprefferred=True, add_persona=False, include_level=include_level)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")
                # print(k, padding_value)
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    # assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    # assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    # assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       include_level: bool = False,
                       exact: bool = False) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, include_level=include_level).items():
                flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode, {"metadata": data.get("metadata")}))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode, extra in flat_data:
            if done:
                break
            if sft_mode:
                if isinstance(sft_target, str):
                    sft_target = [sft_target]
                for s in sft_target:
                    batch_element = tokenize_batch_element(prompt, s, s, truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                    batch.append({**batch_element, **extra})
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'Finished generating {n_examples} examples on {split} split')
                            done = True

                        batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append({**batch_element, **extra})
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1
    
    if exact and len(batch) > 0: ## for generation we need that last bit
        yield collate_fn(batch)
