import wandb
import os
class WandbManager:
    def __init__(self, project_name, swap_dir='./wandb_swap'):
        self.project_name = project_name
        self.swap_dir = swap_dir
        wandb.login()
        if not os.path.exists(self.swap_dir):
            os.makedirs(self.swap_dir)

    def download_model(self, model_name, tag):
        artifact = wandb.use_artifact(f"{model_name}:{tag}")
        artifact.download(target_dir=self.swap_dir)
    

    def upload_model(self, model, model_path, model_name, tag):
        artifact = wandb.Artifact(name=model_name, type='model', description=f"Model artifact for {tag}")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    def log_img_table(self, columns, rows):
        table = wandb.Table(columns=columns)
        for row in rows:
            img = row[0]
            row[0] = wandb.Image(img)
            table.add_data(*row)
        wandb.log({"predictions_table":table}, commit=False)
    
    def log(self, registry:dict):
        wandb.log(registry)
    
    def alert(self, title, text):
        wandb.alert(
            title=title,
            text=text
        )
    
    def start(self, project, config: dict = None):
        if config is None:
            print("WandbManager: starting project ", self.project_name)
            wandb.init(project=self.project_name)
        else:
            print("WandbManager: starting project and config ", self.project_name, config)
            wandb.init(project=self.project_name, config=config)
    
    def stop(self):
        wandb.finish()
    
    def get_config(self):
        return wandb.config

    def start_agent(self, train_function, sweep_id="i1gjnonk", count=1):
        wandb.agent(sweep_id=sweep_id, function=train_function, count=count)

#
# metrics_wandb = WandbManager("PytorchWrapper")