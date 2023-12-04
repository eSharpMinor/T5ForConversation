# Imports
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration  
from transformers import AdamW
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence

MODEL_NAME = "t5-small"
TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-5


class T5Dataset:
    
  def __init__(self,question,answer, TOKENIZER=TOKENIZER, INPUT_MAX_LEN=128, OUTPUT_MAX_LEN=128):   
    
    self.question = question
    self.answer = answer
    self.tokenizer = TOKENIZER
    self.input_max_len = INPUT_MAX_LEN
    self.output_max_len = OUTPUT_MAX_LEN
  
  def __len__(self):                      # This method retrives the number of item from the dataset
    return len(self.question)

  def __getitem__(self,item):             # This method retrieves the item at the specified index item. 

    question = str(self.question[item])
    question = ''.join(question.split())

    answer = str(self.answer[item])
    answer = ''.join(answer.split())

    input_tokenize = self.tokenizer(      
            question,
            add_special_tokens=True,
            max_length=self.input_max_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors="pt"
        )
    output_tokenize = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=self.output_max_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors="pt"   
        )
    

    input_ids = input_tokenize["input_ids"].flatten()
    attention_mask = input_tokenize["attention_mask"].flatten()
    labels = output_tokenize['input_ids'].flatten()

    out = {
            'question':question,      
            'answer':answer,
            'input_ids': input_ids,
            'attention_mask':attention_mask,
            'target':labels
        }
        
    return out
    
class T5DataLoad(pl.LightningDataModule):
    
    def __init__(self, df_train, df_test):
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test
        self.tokenizer = TOKENIZER
        INPUT_MAX_LEN=128
        OUTPUT_MAX_LEN=128
        self.input_max_len = INPUT_MAX_LEN
        self.out_max_len = OUTPUT_MAX_LEN
    
    def setup(self, stage=None):
        
        self.train_data = T5Dataset(
            question = self.df_train.question.values,
            answer = self.df_train.answer.values
        )
        
        self.valid_data = T5Dataset(
            question = self.df_test.question.values,
            answer = self.df_test.answer.values
        )
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
         self.train_data,
         batch_size= TRAIN_BATCH_SIZE,
         shuffle=True, 
         num_workers=2
        )
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
        self.valid_data,
        batch_size= VAL_BATCH_SIZE,
        num_workers = 2
        )
