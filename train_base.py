# Imports
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration  
from transformers import AdamW
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence


from model_base import T5Model, MODEL_NAME
from utils_base import T5Dataset, T5DataLoad


import argparse

import warnings
warnings.filterwarnings("ignore")

def main():
    pl.seed_everything(100)
    parser = argparse.ArgumentParser(description='A t5 pipeline with transformers library')
    parser.add_argument('-g_s', '--global_seed', help='Define seed for reproducability purpose', default=100, type=int)
    parser.add_argument('-d_d', '--data_dir', help='Directory, where the dataset can be found', default=None, type=str)
    parser.add_argument('-p_m', '--pretrained_model', help='Path of the Pretrained model (.bin /.pth)', default=None, type=str)
    parser.add_argument('-i_s_l', '--input_seq_len', help='Maximum length boundary for input sequences', default=128, type=int)
    parser.add_argument('-o_s_l', '--output_seq_len', help='Maximum length boundary for output sequences', default=128, type=int)
    parser.add_argument('-s_c', '--save_checkpoint', help='Whether you want to save the checkpoint yes/no', default='yes', type=str)
    parser.add_argument('-t_b_s', '--train_batch_size', help='Batch size for Training', default=16, type=int)
    parser.add_argument('-e_b_s', '--eval_batch_size', help='Batch size for Evaluation', default=8, type=int)
    parser.add_argument('-l_r', '--learning_rate', help='Learning rate', default=3e-5, type=float)
    parser.add_argument('-n_t_e', '--num_train_epochs', help='Number of training epochs', default=1, type=int)
    parser.add_argument('-n_w_s', '--num_warmup_steps', help='Number of warmup steps', default=0, type=int)
    parser.add_argument('-o_d', '--output_dir', help='Directory for the output file', default=None, type=str)
    args = vars(parser.parse_args())
    
    # Passing arguments to variables
    GLOBAL_SEED = args['global_seed']
    DATA_DIR = args['data_dir']
    PRETRAINED_MODEL = args['pretrained_model']
    INPUT_SEQ_LEN = args['input_seq_len']
    OUTPUT_SEQ_LEN = args['output_seq_len']
    TRAIN_BATCH_SIZE = args['train_batch_size']
    VAL_BATCH_SIZE = args['eval_batch_size']
    LEARNING_RATE = args['learning_rate']
    EPOCHS = args['num_train_epochs']
    NUM_WARMUP_STEPS = args['num_warmup_steps']
    OUTPUT_DIR = args['output_dir']
    SAVE_CHECKPOINT = args['save_checkpoint']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=512)    
    pl.seed_everything(GLOBAL_SEED)
    
    data = pd.read_csv(DATA_DIR)
    data.drop(columns=['Unnamed: 0'],inplace=True)
    
    def run():
    	df_train, df_test = train_test_split(data,test_size = 0.2, random_state=GLOBAL_SEED)
    	dataload = T5DataLoad(df_train,df_test)
    	dataload.setup()
    	device = DEVICE
    	model = T5Model()
    	model.to(device)
    	
    	checkpoint = ModelCheckpoint(
    	    dirpath=OUTPUT_DIR,
    	    filename='best-model',
    	    save_top_k=2,
    	    verbose=True,
    	    monitor="val_loss",
    	    mode="min"
    	    )
    	    
    	trainer = pl.Trainer(
    	    callbacks = checkpoint,
    	    max_epochs= EPOCHS,
    	    devices=1,
    	    accelerator="gpu"
    	    )
    	trainer.fit(model, dataload)
    # Run the training
    run()

if __name__ == "__main__":
    main() 
    
