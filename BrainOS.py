from transformers import pipeline, set_seed
import time
import random


#BrainClock = 10
BrainCycle = 2
BusyMind = 10
generator = pipeline('text-generation', model='gpt2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
set_seed(42)

def State(what_the_world_is_like_now):
    what_the_world_is_like_now = summarizer(what_the_world_is_like_now, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    influences = generator(what_the_world_is_like_now, max_length=130, num_return_sequences=1)[0]['generated_text'] 
    return influences

def HowTheWorldEvolves():
    _my_points_of_view = ''
    with open('my_points_of_view.txt') as f:
        my_points_of_view = str(f.readlines()[0])
    for _ in range(BusyMind):
        _my_points_of_view += random.choice(my_points_of_view.split('.')) 
    my_points_of_view = summarizer(_my_points_of_view, max_length=130, min_length=30, do_sample=False)[0]['summary_text']       
    return my_points_of_view

def WhatMyActionsDo():
    _my_impacts = ''
    with open('my_impacts.txt') as f:
        my_impacts = str(f.readlines()[0])
    for _ in range(BusyMind):    
        _my_impacts += random.choice(my_impacts.split('.')) 
    my_impacts = summarizer(_my_impacts, max_length=130, min_length=30, do_sample=False)[0]['summary_text']         
    return my_impacts

def Utility():
    _my_feelings = ''
    with open('my_feelings.txt') as f:
        my_feelings = str(f.readlines()[0])
    for _ in range(BusyMind):     
        _my_feelings += random.choice(my_feelings.split('.')) 
    my_feelings = summarizer(_my_feelings, max_length=130, min_length=30, do_sample=False)[0]['summary_text']         
    return my_feelings

class BrainOS:
    def __init__(self, BrainClock = 10):
        self.BrainClock = BrainClock
        self.what_the_world_is_like_then = ''

    def PrimaryConsciousness(self, what_the_world_is_like_now):
        influences = State(what_the_world_is_like_now)
        my_points_of_view = HowTheWorldEvolves()
        my_impacts = WhatMyActionsDo()
        what_the_world_is_like_now += influences + my_points_of_view + my_impacts 
        what_the_world_is_like_now = summarizer(what_the_world_is_like_now, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] 
        what_it_will_be_like_if_i_do_an_action = generator(what_the_world_is_like_now, max_length=130, num_return_sequences=1)[0]['generated_text'] 
        return what_it_will_be_like_if_i_do_an_action

    def SecondaryConsciousness(self, what_it_will_be_like_if_i_do_an_action):
        my_points_of_view = HowTheWorldEvolves()
        my_impacts = WhatMyActionsDo() 
        what_it_will_be_like_if_i_do_an_action += my_points_of_view + my_impacts
        what_it_will_be_like_if_i_do_an_action = summarizer(what_it_will_be_like_if_i_do_an_action, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] 
        how_happy_i_will_be_in_such_a_state = generator(what_it_will_be_like_if_i_do_an_action, max_length=130, num_return_sequences=1)[0]['generated_text'] 
        return how_happy_i_will_be_in_such_a_state

    def TertiaryConsciousness(self, how_happy_i_will_be_in_such_a_state):
        my_feelings = Utility()
        how_happy_i_will_be_in_such_a_state += my_feelings
        how_happy_i_will_be_in_such_a_state = summarizer(how_happy_i_will_be_in_such_a_state, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] 
        what_action_I_should_do_now = generator(how_happy_i_will_be_in_such_a_state, max_length=130, num_return_sequences=1)[0]['generated_text'] 
        return what_action_I_should_do_now
    
    def iamlive(self, what_the_world_is_like_now):
        what_the_world_is_like_now += self.what_the_world_is_like_then
        for _ in range(BrainCycle):
            print ('tictoc tictoc ......................................')
            what_it_will_be_like_if_i_do_an_action = self.PrimaryConsciousness(what_the_world_is_like_now)
            how_happy_i_will_be_in_such_a_state = self.SecondaryConsciousness(what_it_will_be_like_if_i_do_an_action)
            what_action_I_should_do_now = self.TertiaryConsciousness(how_happy_i_will_be_in_such_a_state)
            print ('Hearing: ', what_the_world_is_like_now, ', Chain of thoughts: ', what_action_I_should_do_now) 
            #time.sleep(self.BrainClock)
            what_the_world_is_like_now += what_action_I_should_do_now
        self.what_the_world_is_like_then = summarizer(what_the_world_is_like_now, max_length=130, min_length=30, do_sample=False)[0]['summary_text']