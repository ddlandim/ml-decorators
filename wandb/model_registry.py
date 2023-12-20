# Importando as bibliotecas necessárias
import wandb
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Inicializando o projeto W&B
wandb.init(project="seq_shape_10_u")

# Criando um modelo Keras simples
model = Sequential([
    Dense(10, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])

# Salvando o modelo em um arquivo .h5
model_path = 'my_model.h5'
model.save(model_path)

# Simulando o registro de métricas do modelo
wandb.log({"acc": random.random()})

# Criando um Artifact para o modelo
best_model = wandb.Artifact(f"model_{wandb.run.id}", type='model')
best_model.add_file(model_path)

# Registrando o Artifact
wandb.log_artifact(best_model)

# Vinculando o modelo ao registro do modelo
wandb.run.link_artifact(best_model, 'model-registry/My Registered Model')

# Finalizando a execução
wandb.finish()

###
"""
wandb: Currently logged in as: semantix-douglas-diniz. Use `wandb login --relogin` to force relogin
Tracking run with wandb version 0.15.8
Run data is saved locally in /content/wandb/run-20230823_204911-ekpgxc1s
Syncing run sleek-universe-1 to Weights & Biases (docs)
View project at https://wandb.ai/semantix-douglas-diniz/seq_shape_10_u
View run at https://wandb.ai/semantix-douglas-diniz/seq_shape_10_u/runs/ekpgxc1s
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
Waiting for W&B process to finish... (success).
0.016 MB of 0.024 MB uploaded (0.000 MB deduped)
"""