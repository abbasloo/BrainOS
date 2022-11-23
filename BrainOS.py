from transformers import pipeline, set_seed
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import random


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
generator = pipeline('text-generation', model='gpt2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
set_seed(42)
empty = '. '

def InnerDialog(you):
    for step in range(1):
        new_user_input_ids = tokenizer.encode(you + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        me = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return me    
    
def State(what_the_world_is_like_now):
    what_the_world_is_like_now = summarizer(what_the_world_is_like_now, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    influences = generator(what_the_world_is_like_now, max_length=130, num_return_sequences=1)[0]['generated_text'] 
    return influences

def HowTheWorldEvolves(BusyMind, insentence, threshold):
    _my_points_of_view = ''
    with open('my_points_of_view.txt') as f:
        my_points_of_view = str(f.readlines()[0])
    i = 0    
    ii = 0
    choices = my_points_of_view.split('.')
    while i < BusyMind:
        sentence = random.choice(choices) 
        sentences = [sentence, insentence]
        embeddings = similarity.encode(sentences)
        score = sum(embeddings[0]*embeddings[1])
        ii += 1
        if score > threshold:
            _my_points_of_view += empty + sentence
            i += 1
        if ii > 100:
            threshold *= 0.2
            ii = 0
    my_points_of_view = summarizer(_my_points_of_view, max_length=130, min_length=30, do_sample=False)[0]['summary_text']       
    return my_points_of_view

def WhatMyActionsDo(BusyMind, insentence, threshold):
    _my_impacts = ''
    with open('my_impacts.txt') as f:
        my_impacts = str(f.readlines()[0])
    i = 0 
    ii = 0
    choices = my_impacts.split('.')
    while i < BusyMind:
        sentence = random.choice(choices) 
        sentences = [sentence, insentence]
        embeddings = similarity.encode(sentences)
        score = sum(embeddings[0]*embeddings[1])
        ii += 1 
        if score > threshold:
            _my_impacts += empty + sentence
            i += 1       
        if ii > 100:
            threshold *= 0.2   
            ii = 0 
    my_impacts = summarizer(_my_impacts, max_length=130, min_length=30, do_sample=False)[0]['summary_text']         
    return my_impacts

def Utility(BusyMind, insentence, threshold):
    _my_feelings = ''
    with open('my_feelings.txt') as f:
        my_feelings = str(f.readlines()[0])
    i = 0    
    ii = 0
    choices = my_feelings.split('.')
    while i < BusyMind:
        sentence = random.choice(choices) 
        sentences = [sentence, insentence]
        embeddings = similarity.encode(sentences)
        score = sum(embeddings[0]*embeddings[1])
        ii += 1
        if score > threshold:
            _my_feelings += empty + sentence + empty + InnerDialog(sentence) 
            #_my_feelings += empty +summarizer(sentence + empty + InnerDialog(sentence), max_length=130, min_length=30, do_sample=False)[0]['summary_text']  
            i += 1    
        if ii > 100:
            threshold *= 0.2 
            ii = 0
    my_feelings = summarizer(_my_feelings, max_length=130, min_length=30, do_sample=False)[0]['summary_text']         
    return my_feelings

def Memory(BusyMind, mems, insentence, threshold):
    if mems == '':
        return mems
    mymems = ''
    i = 0    
    ii = 0
    choices = mems.split('.')
    while i < BusyMind:
        sentence = random.choice(choices) 
        sentences = [sentence, insentence]
        embeddings = similarity.encode(sentences)
        score = sum(embeddings[0]*embeddings[1])
        ii += 1
        if score > threshold:
            mymems += empty + sentence
            i += 1    
        if ii > 100:
            threshold *= 0.2 
            ii = 0
    return mymems        

class BrainOS:
    def __init__(self, BrainClock = 10, BrainCycle = 5, BusyMind = 20, threshold = 0.5):
        self.BrainCycle = BrainCycle
        self.BrainClock = BrainClock
        self.BusyMind = BusyMind
        self.threshold = threshold
        self.what_the_world_is_like_then = ''
        self.Memory = ''

    def PrimaryConsciousness(self, what_the_world_is_like_now):
        mems = Memory(self.BusyMind, self.Memory , what_the_world_is_like_now, self.threshold)
        influences = State(what_the_world_is_like_now + empty + mems)
        my_points_of_view = HowTheWorldEvolves(self.BusyMind, what_the_world_is_like_now, self.threshold)
        my_impacts = WhatMyActionsDo(self.BusyMind, what_the_world_is_like_now, self.threshold)
        what_the_world_is_like_now += empty + influences + empty + my_points_of_view + empty + my_impacts 
        what_the_world_is_like_now = summarizer(what_the_world_is_like_now, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] 
        what_it_will_be_like_if_i_do_an_action = generator(what_the_world_is_like_now, max_length=130, num_return_sequences=1)[0]['generated_text'] 
        return what_it_will_be_like_if_i_do_an_action

    def SecondaryConsciousness(self, what_it_will_be_like_if_i_do_an_action):
        my_points_of_view = HowTheWorldEvolves(self.BusyMind, what_it_will_be_like_if_i_do_an_action, self.threshold)
        my_impacts = WhatMyActionsDo(self.BusyMind, what_it_will_be_like_if_i_do_an_action, self.threshold) 
        what_it_will_be_like_if_i_do_an_action += empty + my_points_of_view + empty + my_impacts
        what_it_will_be_like_if_i_do_an_action = summarizer(what_it_will_be_like_if_i_do_an_action, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] 
        how_happy_i_will_be_in_such_a_state = generator(what_it_will_be_like_if_i_do_an_action, max_length=130, num_return_sequences=1)[0]['generated_text'] 
        return how_happy_i_will_be_in_such_a_state

    def TertiaryConsciousness(self, how_happy_i_will_be_in_such_a_state):
        my_feelings = Utility(self.BusyMind, how_happy_i_will_be_in_such_a_state, self.threshold)
        dialogs = how_happy_i_will_be_in_such_a_state.split('.')
        how_happy_i_will_be_in_such_a_state = ''
        for d in dialogs:
            how_happy_i_will_be_in_such_a_state += empty + d + empty + InnerDialog(d)   
            #how_happy_i_will_be_in_such_a_state = summarizer(how_happy_i_will_be_in_such_a_state, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] 
        how_happy_i_will_be_in_such_a_state += empty + my_feelings
        how_happy_i_will_be_in_such_a_state = summarizer(how_happy_i_will_be_in_such_a_state, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] 
        what_action_I_should_do_now = generator(how_happy_i_will_be_in_such_a_state, max_length=130, num_return_sequences=1)[0]['generated_text'] 
        return what_action_I_should_do_now
    
    def iamlive(self, what_the_world_is_like_now):
        if len(self.Memory) > 1:
            self.Memory = summarizer(self.Memory, max_length=30, min_length=10, do_sample=False)[0]['summary_text']
        what_the_world_is_like_now += empty + self.what_the_world_is_like_then
        for _ in range(self.BrainCycle):
            what_it_will_be_like_if_i_do_an_action = self.PrimaryConsciousness(what_the_world_is_like_now)
            how_happy_i_will_be_in_such_a_state = self.SecondaryConsciousness(what_it_will_be_like_if_i_do_an_action)
            what_action_I_should_do_now = self.TertiaryConsciousness(how_happy_i_will_be_in_such_a_state)
            print ('Am thinking ......................................')
            print ('Hearing: ', what_the_world_is_like_now, ' Chain of thoughts: ', what_action_I_should_do_now) 
            print ('Am aware ......................................')            
            #time.sleep(self.BrainClock)
            what_the_world_is_like_now = what_action_I_should_do_now
            self.Memory += empty + summarizer(what_action_I_should_do_now, max_length=30, min_length=10, do_sample=False)[0]['summary_text']
        self.what_the_world_is_like_then = summarizer(what_the_world_is_like_now, max_length=30, min_length=10, do_sample=False)[0]['summary_text']
