# Importando as bibliotecas necessárias
import wandb
from tensorflow.keras.models import load_model
import numpy as np

# Nome do projeto e ID do artefato
project_name = "seq_shape_10_u"
artifact_name = "model_ID_DO_ARTEFATO" # Substitua com a ID do artefato

# Inicializando o projeto W&B
wandb.init(project=project_name)

# Baixando o artefato do modelo
artifact = wandb.use_artifact(f"{artifact_name}:latest")
artifact_dir = artifact.download()

# Carregando o modelo
model_path = artifact_dir + '/my_model.h5'
model = load_model(model_path)

# Criando dados de entrada de exemplo para predição (adequado ao tamanho da entrada do modelo)
input_data = np.random.rand(1, 10)

# Fazendo predições
predictions = model.predict(input_data)

# Você pode agora usar as predições conforme necessário
print("Predições:", predictions)

# Finalizando a execução
wandb.finish()
