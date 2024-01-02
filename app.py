from transformers import T5Tokenizer
from model_base import T5Model

import torch
import streamlit as st
import random
import time

MODEL_NAME="t5-base"
INPUT_MAX_LEN=64
MODEL_MAX_LENGTH=512
CHECKPOINT="output_conv_10/output_base_conv_5/best-model.ckpt"
prompt = None

#@st.cache_resource
#def get_model(CHECKPOINT):
#    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    global TOKENIZER
#    TOKENIZER = T5Tokenizer.from_pretrained("t5-base", model_max_length=model_max_length)
#    with torch.no_grad():
#        global MODEL
#        MODEL = T5Model.load_from_checkpoint(CHECKPOINT, device=devices)
#        MODEL.freeze()
        
@st.cache_resource
def load_model(CHECKPOINT):
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = T5Model.load_from_checkpoint(CHECKPOINT, device=devices)
    MODEL.eval()
    return MODEL
    
@st.cache_resource
def load_tokenizer(MODEL_NAME, MODEL_MAX_LENGTH):
    TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=MODEL_MAX_LENGTH)
    return TOKENIZER

def generate_question(question, model, tokenizer):

    inputs_encoding =  tokenizer(
        question,
        add_special_tokens=True,
        max_length= INPUT_MAX_LEN,
        padding = 'max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_tensors="pt"
        )

    
    generate_ids = model.model.generate(
        input_ids = inputs_encoding["input_ids"],
        attention_mask = inputs_encoding["attention_mask"],
        max_length = INPUT_MAX_LEN,
        num_beams = 4,
        num_return_sequences = 1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        )
    
    generate_ids = generate_ids[0][2:-1]
    preds = tokenizer.convert_ids_to_tokens(generate_ids)
    #preds = [
    #    tokenizer.decode(gen_id,
    #    skip_special_tokens=True, 
    #    clean_up_tokenization_spaces=True)
    #    for gen_id in generate_ids
    #]
    return " ".join(preds).capitalize()
    #return generate_ids

MODEL = load_model(CHECKPOINT)
TOKENIZER = load_tokenizer(MODEL_NAME, MODEL_MAX_LENGTH)

st.title("Conversational Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
if prompt != None:
    response = generate_question(prompt, MODEL, TOKENIZER)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
    	st.markdown(response)
    	# Add assistant response to chat history
    	st.session_state.messages.append({"role": "assistant", "content": response})
