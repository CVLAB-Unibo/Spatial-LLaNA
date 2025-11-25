import argparse
import torch
from torch.utils.data import DataLoader
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)
from spatial_llana.conversation import conv_templates, SeparatorStyle
from spatial_llana.utils import disable_torch_init
from spatial_llana.model import Spatial_LLaNA
from spatial_llana.model.utils import KeywordsStoppingCriteria
from spatial_llana.data import Spatial_EvaluationDataset
from spatial_llana.data.utils import DataCollatorForSpatialDataset_Eval
from tqdm import tqdm
from transformers import AutoTokenizer
import transformers
from spatial_llana.train.train_spatial_llana import DataArguments_Eval

from spatial_llana import conversation as conversation_lib
from pathlib import Path

import json
import glob

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Spatial_LLaNA.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.float16).to(args.device)
    model.initialize_tokenizer_weights2space_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    return model, tokenizer, conv

def load_dataset(data_path, data_folder, anno_folder, split, text_data, tokenizer, data_args):

    dataset = Spatial_EvaluationDataset(
        split=split,
        root = data_path,
        data_folder = data_folder,
        anno_folder = anno_folder,
        conversation_type=text_data,
        tokenizer=tokenizer,
        data_args=data_args)

    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4, collate_fn=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader


def run_inference(model, tokenizer, conv, dataloader, output_file_path):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    model.eval()
    responses = []
    max_new_tokens = 512  # Maximum number of new tokens to generate
    for batch in tqdm(dataloader):
        object_ids = batch["object_id"]  # List of object IDs
        input_ids = batch["input_ids"].to(model.device)         # * tensor of B, L
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        vecs = batch["vecs"].to(model.device).to(model.dtype)   # * tensor of B, N, C(3)

        model.eval() 
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                vecs=vecs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[stopping_criteria]) # * B, L'

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs = [output.strip() for output in outputs]

        # Get conversations from batch (required for evaluation with ground truth)
        conversations_batch = batch.get("conversations", None)
        if conversations_batch is None:
            raise ValueError("Conversations are required in the batch for evaluation. "
                           "The dataset must return conversations to extract ground truth answers.")
        
        # Iterate through outputs and corresponding batch items
        for idx, output in enumerate(outputs):
            if isinstance(object_ids, list):
                obj_id = object_ids[idx]
            else:
                obj_id = object_ids[idx].item() if hasattr(object_ids[idx], 'item') else object_ids[idx]
            
            # Get conversations for this batch item
            convs = conversations_batch[idx] if isinstance(conversations_batch, list) else conversations_batch
            
            for i in range(0, len(convs), 2):
                if convs[i]["from"]=="human":
                    q = convs[i]["value"].replace("<point>\n", "").strip()
                else:
                    raise Exception('The first conversation should be from human.')
                if convs[i+1]["from"]=="gpt":
                    gt_a = convs[i+1]["value"].strip()
                else:
                    raise Exception('The second conversation should be from gpt.')
                responses.append({
                    "object_id": obj_id,
                    "question": q,
                    "ground_truth": gt_a,
                    "model_output": output
                })
                print('\nquestion:', q)
                print('ground_truth: ', gt_a)
                print('result: ', output)
                print('= = = = = = = = = = = = = = = = = = = = = = = = = = =')
    

    os.makedirs(Path(output_file_path).parent, exist_ok=True)
    # save the results to a JSON file
    with open(output_file_path, 'w') as fp:
        json.dump(responses, fp, indent=2)

    # * print info
    print(f"Saved results to {output_file_path}")

    return responses


def main(args):
    print('*** model_name:', args.model_name)
    print('**** model_name split:', args.model_name.split('/')[-1])
    output_folder = os.path.join(args.output_dir, args.model_name.split('/')[-1])

    print('**** output_folder:', output_folder)
        
    os.makedirs(output_folder, exist_ok=True)
    output_filename = f'{args.split}.json'

    output_file_path = os.path.join(output_folder, output_filename)
    args.device = torch.device(f'cuda:{args.device}')

    if os.path.exists(output_file_path):
        print(f'[INFO] {output_file_path} already exists.')
        return

    model, tokenizer, conv = init_model(args)
    weights2space_config = model.get_model().weights2space_config

    # Create DataArguments_Eval directly without re-parsing command line
    data_args = DataArguments_Eval()

    # Set the required fields from weights2space_config
    data_args.point_token_len = weights2space_config['point_token_len']
    data_args.mm_use_point_start_end = weights2space_config['mm_use_point_start_end']
    data_args.weights2space_config = weights2space_config
    

    dataset = load_dataset(args.data_path, args.data_folder, args.anno_folder, args.split, args.text_data, tokenizer, data_args)
    data_collator = DataCollatorForSpatialDataset_Eval(tokenizer=tokenizer)
    dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers, data_collator)
    
    print(f'[INFO] Start generating results for {output_file_path}.')
    results = run_inference(model, tokenizer, conv, dataloader, output_file_path)

    # * release model and tokenizer, and release cuda memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="andreamaduzzi/Spatial-LLaNA-13B", help="Name of the model to evaluate.") 
    parser.add_argument("--device", type=int, default=0, help="idx of the GPU to use")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save the evaluation results.")

    # * dataset type
    parser.add_argument("--data_path", type=str, default="data/spatial_llana_dataset", required=False)

    parser.add_argument("--data_folder", default="vecs", type=str, help="Name of folder with embeddings.")  # vecs_resize to use the embeds computed on NeRFs fitted on resized images
    parser.add_argument("--anno_folder", default="texts", type=str, help="Name of folder with conversations.")
    parser.add_argument("--split", type=str, default="pointllm_test", choices=["shapenerf_test", "hst", "objanerf_pointllm_test", "objanerf_gpt4point_test", "spatial_objanerf"], required=False)
    parser.add_argument("--text_data", type=str, default="brief_description", choices=["brief_description", "detailed_description", "single_round"], required=False, help="Type of text data to use for inference of the test set of ShapeNeRF-Text.")

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    args = parser.parse_args()

    main(args)