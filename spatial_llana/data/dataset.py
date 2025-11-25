from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
import copy
from .utils import preprocess_embedding, preprocess_v1_eval, DataCollatorForSpatialDataset
from transformers import PreTrainedTokenizer

class Spatial_Dataset(Dataset):
    """Dataset utilities for NeRF-text dataset."""
    def __init__(self,
                 root=None,           # root of the dataset
                 data_folder=None,    # name of the folder with <model_id>.npy
                 anno_folder=None,    # name of the folder with JSON containing conversations for each model_id
                 tokenizer=None,
                 split='train',
                 conversation_types=None, # * default is simple_des, used for stage1 pre-train
                 data_args=None):

        """
        split: data split.
        conversation_types: tuple, used to filter the data, default is ('brief_description'), other types is:
            "detailed_description", "single_round", "multi_round".
        tokenizer: load point clouds only if None
        """
        super(Spatial_Dataset, self).__init__()

        """Initialize dataset with object point clouds and text"""
        self.root = root
        self.data_folder = data_folder
        self.anno_folder = anno_folder
        self.tokenizer = tokenizer
        self.split = split 
        if conversation_types is None:
            self.conversation_types = ["brief_description"]
        else:
            self.conversation_types = conversation_types

        self.data_args = data_args
        print('data_args:', data_args)
        self.weights2space_config = data_args.weights2space_config if data_args is not None else None
        self.vec_indicator = '<point>'
        self.data_path = os.path.join(self.root, self.split, self.data_folder)  # path to vecs
        
        # read text annotations
        print(f'= = = = = = = = conversation_types: {self.conversation_types} = = = = = = = =')

        self.list_data_dict = []
        # Load brief descriptions if specified
        if "brief_description" in self.conversation_types:
            brief_anno_path = os.path.join(self.root, self.split, self.anno_folder, 'conversations_brief.json')
            print(f"Loading brief descriptions from {brief_anno_path}.")
            with open(brief_anno_path, "r") as json_file:
                self.list_data_dict.extend(json.load(json_file))

        # Load complex conversations if specified
        complex_types = {"detailed_description", "single_round", "multi_round"}
        if any(conv_type in self.conversation_types for conv_type in complex_types):
            complex_anno_path = os.path.join(self.root, self.split, self.anno_folder, 'conversations_complex.json')
            print(f"Loading complex conversations from {complex_anno_path}.")
            with open(complex_anno_path, "r") as json_file:
                self.list_data_dict.extend(json.load(json_file))
        
        # * extract the desired conversations, belonging to conversation_types
        self.list_data_dict = [data for data in self.list_data_dict if data['conversation_type'] in self.conversation_types ]

        # * print the size of different conversation_type
        for conversation_type in self.conversation_types:
            print(f"Number of {conversation_type}: {len([data for data in self.list_data_dict if data['conversation_type'] == conversation_type])}")
 
    def _load_vec(self, object_id):
        filename = f"{object_id}.npy"  
        vec = np.load(os.path.join(self.data_path, filename))
        return vec
    
    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "sources should be a list"
        if self.vec_indicator in sources[0]['conversations'][0]['value']:

            object_id = self.list_data_dict[index]['object_id']

            vec = self._load_vec(object_id) # shape: (1024,)
            if len(vec.shape) == 1:         # If I get a single token, I unsqueeze it to (1, 1024)
                vec = np.expand_dims(vec, axis=0)
            if self.tokenizer is None:
                data_dict = dict(
                    vecs=torch.from_numpy(vec.astype(np.float32)),
                    object_ids=object_id
                )
                return data_dict

            sources = preprocess_embedding(    # using the same code as PointLLM to add placeholders for vecs inside the text
                copy.deepcopy([e["conversations"] for e in sources]), self.weights2space_config, point_indicator=self.vec_indicator)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess_v1(  # data processing on conversations
            sources,
            self.tokenizer)

        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # vecs exist in the data
        if self.vec_indicator in self.list_data_dict[index]['conversations'][0]['value']:
            data_dict['vecs'] = torch.from_numpy(vec.astype(np.float32))

        return data_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.list_data_dict)

def make_spatial_data_module(tokenizer: PreTrainedTokenizer, data_args):
    """Make dataset and collator for Joint3Ddataset with text and point cloud data."""
    """Initialize datasets."""

    data_collator = DataCollatorForSpatialDataset(tokenizer=tokenizer)
    print("Loading training datasets.")
    train_dataset = Spatial_Dataset(
        split='train', # * load train split
        root=data_args.root,
        data_folder=data_args.data_folder,
        anno_folder=data_args.anno_folder,
        conversation_types=data_args.conversation_types,
        tokenizer=tokenizer,
        data_args=data_args
    )
    print("Done!")
    
    # * make a val dataset
    print("Loading validation datasets.")
    val_dataset = Spatial_Dataset(
        split='val', # * load train split
        root=data_args.root,
        data_folder=data_args.data_folder,
        anno_folder=data_args.anno_folder,
        conversation_types=data_args.conversation_types,
        tokenizer=tokenizer,
        data_args=data_args
    )
    return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)

class Spatial_EvaluationDataset(Dataset):
    """Dataset utilities for NeRF-text dataset, used for evaluation.
    The difference between this dataset and the original one is that this one provides the input_ids of the question (without the answer) and it provides each 
    question from multi_round conversation separately."""
    
    def __init__(self,
                 root=None,           # root of the dataset
                 data_folder=None,    # name of the folder with <model_id>.npy
                 anno_folder=None,    # name of the folder with JSON containing conversations for each model_id
                 tokenizer=None,
                 split='train',
                 conversation_type=None, # * default is simple_des, used for stage1 pre-train
                 data_args=None):

        """
        split: only considered when data_args.split_train_val is True.
        conversation_type: tuple, used to filter the data, default is ('brief_description'), other types is:
            "detailed_description", "single_round", "multi_round".
        tokenizer: load point clouds only if None
        """
        super(Spatial_EvaluationDataset, self).__init__()

        """Initialize dataset with object point clouds and text"""
        self.root = root
        self.data_folder = data_folder
        self.anno_folder = anno_folder
        self.tokenizer = tokenizer
        self.split = split 
        if conversation_type is None:
            self.conversation_type = "brief_description"
        else:
            self.conversation_type = conversation_type

        self.data_args = data_args
        print('data_args:', data_args)
        self.weights2space_config = data_args.weights2space_config if data_args is not None else None
        self.vec_indicator = '<point>'
        self.data_path = os.path.join(self.root, self.split, self.data_folder)

        # Load annotations based on split
        if 'shapenerf_test' in self.split:
            if self.conversation_type == "brief_description":
                anno_path = os.path.join(self.root, self.split, self.anno_folder, 'conversations_brief.json')
                print(f"Loading anno file from {anno_path}.")
                with open(anno_path, 'r') as fp:
                    self.annos = json.load(fp)
            elif self.conversation_type == "detailed_description" or self.conversation_type == "single_round":
                anno_path = os.path.join(self.root, self.split, self.anno_folder, 'conversations_complex.json')
                print(f"Loading anno file from {anno_path}.")
                with open(anno_path, 'r') as fp:
                    self.annos = json.load(fp)
                    self.annos = [item for item in self.annos if item['conversation_type'] == self.conversation_type]
        elif 'hst' in self.split:
            anno_path = os.path.join(self.root, 'hst_dataset_filtered.json')
            print(f"Loading anno file from {anno_path}.")
            with open(anno_path, 'r') as fp:
                self.annos = json.load(fp)
            self.data_path = os.path.join(self.root, 'shapenerf_test', self.data_folder)
        elif self.split.startswith('objanerf_'):
            anno_path = os.path.join(self.root, self.split, self.anno_folder, 'conversations_brief.json')
            print(f"Loading anno file from {anno_path}.")
            with open(anno_path, 'r') as fp:
                self.annos = json.load(fp)
        elif self.split == 'spatial_objanerf':
            anno_path = os.path.join(self.root, self.split, self.anno_folder, 'spatial_descriptions.json')
            print(f"Loading anno file from {anno_path}.")
            with open(anno_path, 'r') as fp:
                self.annos = json.load(fp)

 
    def _load_vec(self, object_id):
        filename = f"{object_id}.npy"  
        vec = np.load(os.path.join(self.data_path, filename))
        return vec
    
    def __getitem__(self, index):
        sources = self.annos[index]
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "sources should be a list"
        if self.vec_indicator in sources[0]['conversations'][0]['value']:

            object_id = self.annos[index]['object_id']

            vec = self._load_vec(object_id)
            if len(vec.shape) == 1:         
                vec = np.expand_dims(vec, axis=0)
            if self.tokenizer is None:
                data_dict = dict(
                    vecs=torch.from_numpy(vec.astype(np.float32)),
                    object_ids=object_id
                )
                return data_dict

            sources = preprocess_embedding(    # using the same code as PointLLM to add placeholders for vecs inside the text
                copy.deepcopy([e["conversations"] for e in sources]), self.weights2space_config, point_indicator=self.vec_indicator)
        else:
            print('NOT adding vecs...')
            sources = copy.deepcopy([e["conversations"] for e in sources])

        
        data_dict = preprocess_v1_eval(  # data processing on conversations
            sources,
            self.tokenizer)

        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             object_id=self.annos[index]['object_id'])

        # vecs exist in the data
        if self.vec_indicator in self.annos[index]['conversations'][0]['value']:
            data_dict['vecs'] = torch.from_numpy(vec.astype(np.float32))

        data_dict["object_id"] = self.annos[index]['object_id']
        # Add conversations to data_dict for ground truth extraction during evaluation
        data_dict["conversations"] = self.annos[index]['conversations']
        return data_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)