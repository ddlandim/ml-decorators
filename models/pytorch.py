import math
import torch
import random
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as T
import wandb
class PytorchWrapper:
    @staticmethod
    def torchvision_mnist_dataloader(is_train, batch_size, slice=5):
        full_dataset = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
        sub_dataset = torch.utils.data.Subset(full_dataset, indices=range(0, len(full_dataset), slice))
        loader = torch.utils.data.DataLoader(dataset=sub_dataset,
                                             batch_size=batch_size,
                                             shuffle=True if is_train else False,
                                             pin_memory=True, num_workers=2)
        return loader

    def __init__(self, model_name = 'mnist',
                       executor = 'local',
                       metrics = None,
                       registry = None,
                       config: dict = {}):
        self.model_name = model_name
        self.executor = executor
        self.metrics = metrics
        self.registry = registry
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.init_config(config)
        self.init_model()
        self.train_dl = None
        self.val_dl = None
    
    def init_model(self):
        self.model = self.registry.create_pytorch_seq_hf(self.config)

    def download_model(self):
        self.model = self.registry.download_model(self.model_name)
        print( "checking model parameters, dropout : ", str(self.model.model[4].p))

    def push_model(self):
        self.registry.push_model(self.model, self.model_name, self.config)

    def set_optimizer(self, optimizer:str = None, lr:float = None):
        optimizer = optimizer or self.config.get("optimizer")
        lr = lr or self.config.get("lr")

        default = torch.optim.Adam(self.model.parameters(), lr=lr)

        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer.lower() == "sdg":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            self.optimizer = default
    
    def set_loss_func(self, loss_func:str = None):
        loss_func = loss_func or self.config.get("loss_func")
        default = nn.CrossEntropyLoss()
        if loss_func.lower() == "crossentropyloss":
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = default
    
    def init_config(self,
                    config: dict):
        
        self.config = {
            "arch": "sequential",
            "input_size": config.get("input_size",28 * 28),
            "hidden_size": config.get("hidden_size",256),
            "output_size": config.get("output_size",10),
            "dropout": config.get("dropout",0.5),
            "batch_norm": config.get("batch_norm",True),
            "epochs": config.get("epochs",10),
            "train_batch_size": config.get("train_batch_size",128),
            "val_batch_size": config.get("val_batch_size",256),
            "lr": config.get("lr",1e-3),
            "train_dataset" : config.get("train_dataset","mnist"),
            "loss_func" : config.get("loss_func","crossentropyloss"),
            "optimizer" : config.get("optimizer","adam"),
            "acc_threshold_alert" : config.get("acc_threshold_alert",0.5),
            "dataset_name" : config.get("dataset_name","mnist"),
            "train_split_name" : config.get("train_split_name","train"),
            "val_split_name" : config.get("val_split_name","test")
        }

    def set_dataloaders(self, train_dl, val_dl):
        self.train_dl = train_dl
        self.val_dl = val_dl

    def download_dataloaders(self, dataset_name=None,
                                   train_split_name=None, train_batch_size=None,
                                   val_split_name=None, val_batch_size=None):
        config = self.config
        print("STARTING DATALOADER")
        if self.train_dl is None or self.config.get("dataset_name") != dataset_name:
            self.train_dl = self.registry.get_dataloader(
                                                        dataset_name=dataset_name or config.get("dataset_name"),
                                                        dataset_split_name = train_split_name or config.get("train_split_name"),
                                                        batch_size = train_batch_size or config.get("batch_size"))
            self.val_dl = self.registry.get_dataloader(
                                                        dataset_name=dataset_name or config.get("dataset_name"),
                                                        dataset_split_name = val_split_name or config.get("val_split_name"),
                                                        batch_size = val_batch_size or config.get("batch_size"))
            
            print("train_dl ", self.train_dl, type(self.train_dl))
            print("val_dl ", self.val_dl, type(self.val_dl))

    def log_image_table(self, images, predicted, labels, probs):
        print("LOG IMAGE TABLE")
        columns = ["image", "pred", "target"]+[f"score_{i}" for i in range(10)]
        rows = []
        for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
            rows.append([(img[0].numpy()*255), pred, targ, *prob.numpy()])
        self.metrics.log_img_table(columns, rows)
    
    def validate_model(self,log_images=False, batch_idx=0):
        print("VALIDADE MODEL")
        model = self.model
        valid_dl = self.valid_dl
        self.model.eval()
        device = self.device
        val_loss = 0.
        loss_func = self.loss_func
        with torch.inference_mode():
            correct = 0
            for i, (images, labels) in enumerate(valid_dl):
                images, labels = images.to(device), labels.to(device)

                # Forward pass ➡
                outputs = model(images)
                val_loss += loss_func(outputs, labels)*labels.size(0)

                # Compute accuracy and accumulate
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Log one batch of images to the dashboard, always same batch_idx.
                if i==batch_idx and log_images:
                    self.log_image_table(images, predicted, labels, outputs.softmax(dim=1))
        return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

    def start_sweep_agent(self, sweep_id, count=1):
        self.metrics.start_agent(
            train_function = self.sweep_train,
            sweep_id = sweep_id,
            count = count
        )
    
    def sweep_train(self,config=None):
        print("PytorchWrapper starting")
        with  wandb.init(config=config):
            config = wandb.config
            print("sweep_train_config ", config)
            self.init_config(config)
            self.init_model()
            self.train_model(start=False)
    
    def train_model(self, metrics_init = True):
        if metrics_init:
            self.metrics.start(self.config)
        
        print("train_model started with config ", self.config)
        self.set_optimizer()
        self.set_loss_func()
        #self.download_dataloaders()
        self.train_dl = PytorchWrapper.torchvision_mnist_dataloader(is_train=True, batch_size=self.config.get("train_batch_size"))
        self.valid_dl = PytorchWrapper.torchvision_mnist_dataloader(is_train=False, batch_size=self.config.get("val_batch_size"))
        self.set_optimizer()
        self.set_loss_func()
        n_steps_per_epoch = math.ceil(len(self.train_dl.dataset) / self.config.get("train_batch_size"))
        # Training
        example_ct = 0
        step_ct = 0
        for epoch in range(self.config.get("epochs")):
            print("epoch started with config ", self.config)
            print("train_dl ", self.train_dl, type(self.train_dl))
            print("val_dl ", self.val_dl, type(self.val_dl))
            self.model.train()
            print("self.train_dl type ", self.train_dl)
            for step, (images, labels) in enumerate(self.train_dl):
                
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model.model(images)
                train_loss = self.loss_func(outputs, labels)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                example_ct += len(images)
                metrics = {"train/train_loss": train_loss,
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                        "train/example_ct": example_ct}

                if step + 1 < n_steps_per_epoch:
                    # Log train metrics to wandb
                    self.metrics.log(metrics)

                step_ct += 1

            val_loss, accuracy = self.validate_model(log_images=(epoch==(self.config.get("epochs")-1)))

            # Log train and validation metrics to wandb
            val_metrics = {"val/val_loss": val_loss,
                        "val/val_accuracy": accuracy}
            self.metrics.log({**metrics, **val_metrics})

            print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")
            acc_threshold_alert = self.config.get("acc_threshold_alert")
            if accuracy <= acc_threshold_alert:
                # 🐝 Send the wandb Alert
                self.metrics.alert(
                    title='Low Accuracy',
                    text=f'Accuracy {accuracy} at epoch {epoch} is below the acceptable theshold, {acc_threshold_alert}',
                )
                print('Alert triggered')
                break
        self.metrics.stop()