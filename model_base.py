# Imports
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration  
from transformers import AdamW
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence

MODEL_NAME = "t5-base"
LEARNING_RATE = 3e-5
class T5Model(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict = True)

        
    def forward(self, input_ids, attention_mask, labels=None):
        
        output = self.model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        labels=labels
        )
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["target"]
        loss, logits = self(input_ids , attention_mask, labels)

        
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["target"]
        loss, logits = self(input_ids, attention_mask, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return {'val_loss': loss}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)
