from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForPreTraining, TrainingArguments, Trainer
from huggingface_hub import login as hf_login
from huggingface_hub import HfApi
from huggingface_hub import create_repo
import torchvision.transforms as T
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import json
import os

class PyTorchSeqFH(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dict= {
                        "arch": "sequential",
                        "input_size": 28 * 28,
                        "hidden_size": 256,
                        "output_size": 10,
                        "dropout": 0.5,
                        "batch_norm": True}):
        super(PyTorchSeqFH, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.get("input_size"), config.get("hidden_size")),
            nn.BatchNorm1d(config.get("hidden_size")),
            nn.ReLU(),
            nn.Dropout(config.get("dropout")),
            nn.Linear(config.get("hidden_size"), config.get("output_size"))
        )
        
    def forward(self, x):
        return self.model(x)

class HfacesManager:
    def __init__(self, user_id, hf_token, swap_dir = "./hfaces_swap"):
        self.hf_token = hf_token
        hf_login(token=self.hf_token)
        if not os.path.exists(swap_dir):
            os.makedirs(swap_dir)
        self.swap_dir = swap_dir
        self.api = HfApi()
        self.user_id = user_id

    def get_dataloader(self, dataset_name, dataset_split_name, batch_size,
                       slice=5, shuffle=False, pin_memory= True, num_workers=2):
        
        print("HfacesManager: starting dataloader")
        print("dataset_name ", dataset_name)
        print("dataset_split_name ", dataset_split_name)
        print("batch_size ", batch_size)
        dataset = load_dataset(dataset_name)

        def __transform_to_tensor(example_batch):
            example_batch['image'] = T.ToTensor()(example_batch['image'])
            return example_batch

        dataset = dataset.map(__transform_to_tensor)

        full_dataset = dataset[dataset_split_name]

        split_dataset = [full_dataset[i] for i in range(0, len(full_dataset), slice)]
        split_dataloader = DataLoader(split_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=2)

        #sample_batch = next(iter(split_dataloader))
        #print("Sample Batch:", sample_batch)

        return split_dataloader

    def create_repo(self, repo_name, repo_type, private=False):
        self.repo_url = self.api.create_repo(
                             repo_id = self.user_id + "/" + repo_name,
                             token=self.hf_token,
                             repo_type = repo_type,
                             private=private,
                             exist_ok = True)
        return self.repo_url

    def push_model(self, model, model_name, config = None, branch = None, commit_msg = "Pushing to the hub"):
        
        # Criando um novo repositório no Hugging Face se não existir
        self.create_repo(repo_name=model_name, repo_type="model")
        repo_id = f"{self.user_id}/{model_name}"

        # Salvar o arquivo config.json no mesmo diretório
        model_config = config or model.config
        
        # Empurrar o modelo e o config para o Hugging Face Hub
        if branch is None:
            model.push_to_hub(repo_id=repo_id, config=model_config, commit_message = commit_msg)
        else:
            model.push_to_hub(repo_id=repo_id, branch=branch, config=model_config, commit_message = commit_msg)

    def download_model(self, model_name, branch = None):
        repo_id = f"{self.user_id}/{model_name}"
        model = PyTorchSeqFH.from_pretrained(repo_id)
        return model
    
    @staticmethod
    def create_pytorch_seq_hf(config = None):
        if config is not None:
            model = PyTorchSeqFH(config=config)
        else:
            model = PyTorchSeqFH()
        return model
    
# hfaces = HfacesManager(user_id="ddlandim-semantix", hf_token = "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")