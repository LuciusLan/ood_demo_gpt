from datetime import datetime
import time
import json
import sys
import random
import pdb 
from pathlib import Path
import pickle
import re

from tqdm import tqdm
import openai

# SET OPENAI API KEY

MODEL = 'gpt-3.5-turbo'

random.seed(2023)

pick_sample = random.sample(list(range(5900)), 300)


def completePrompt(p, model, instruction):
    response = openai.Completion.create(
        model=model,
        prompt = instruction + p + "\n\n",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return(response.choices[0].text)

def doQuery(p, model, instruction, ans):
    sysout = completePrompt(p, model, instruction)
    sysout = sysout.strip()
    print(p, "System:", sysout)
    sysout = sysout[:len(ans)]
    return(sysout == ans)

def chatCompletPrompt(p, model):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You are to be asked to answer on what sentiment is expressed in a sentence."},
                {"role": "user", "content": p}
            ],
            max_tokens=5,
            request_timeout=30
        )
    except TypeError:
        print()
    except openai.error.Timeout:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You are to be asked to answer on what sentiment is expressed in a sentence."},
                {"role": "user", "content": p}
            ],
            max_tokens=5,
            request_timeout=30
        )
    except openai.error.RateLimitError:
        print()
    #print(response.choices[0].message)
    return response.choices[0].message.content

def doChatQuery(p, model, ans):
    time.sleep(0.5)
    sysout = chatCompletPrompt(p, model)
    sysout = sysout.strip()
    #print(p, "System:", sysout)
    try:
        sysout = re.search(r'Label:\s?(\w+)', sysout, re.I).groups()[0]
    except TypeError:
        sysout = 'x'
    return(sysout == ans)

# results File: runID - # dd/mm/YY H:M:S
dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")

# CREATE RESULTS FILE
resFile = Path("runs") / dt_string
resFile.touch(exist_ok=False)
resFile.write_text("File,Iteration,Total,VPE Correct,NO VPE Correct\n")

def label2text(label):
    # 0 for neg, 1 for neutral, 2 for pos
    if label == '0\n':
        return 'negative'
    elif label == '1\n':
        return 'neutral'
    elif label == '2\n':
        return 'positive'

# RUN ITERATIONS
for iteration in range(1):
    print("STARTING ITERATION", iteration, "="*30)

    # RUN THROUGH EXAMPLE FILES
        
    with open('./data/rand/book2beauty_rand.json', 'r') as source:
        examples = json.load(source)

        print("Source: BOOK, Target: BEAUTY")

        correct = 0

        # RUN THROUGH EXAMPLES
        for i in tqdm(pick_sample, total=len(pick_sample)):
            e = examples[i]
            prefix = f"Given the input sentence, assign a sentiment label from ['positive', 'neutral', 'negative']. Give your response with the answer label only. Do not include irrelevant text. \nFollowing are a few demonstrations on how to assign the sentiment label:\n"
            demo_text = ''
            for demo in e['source']:
                demo_text += f'Sentence: {demo["s_sent"]}Label:{ label2text(demo["s_label"])}\n'
            prompt = prefix + demo_text + f"Now assign sentiment label for the following sentence. Start your answer with \"Label:\". Sentence:\n{e['q_sent']}"
            answer = label2text(e['q_label'])
            #res = doQuery(prompt, ARGS.model, instructions, answer)
            res = doChatQuery(prompt, MODEL, answer)

            if res is True:
                correct += 1

        with resFile.open("a") as f:
            f.write(f'Accuracy: {correct/300}')
    print(f"iteration {iteration}")
