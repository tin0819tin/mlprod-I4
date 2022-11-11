# app.py
import lightning as L
from lightning.app.components.training import LightningTrainerScript

# run script that trains Surprise with the Lightning Trainer
model_script = 'train.py'
component = LightningTrainerScript(
   model_script,
   num_nodes=2,
   cloud_compute=L.CloudCompute("cpu")
)
app = L.LightningApp(component)