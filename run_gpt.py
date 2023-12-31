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
from openai import OpenAI

client = OpenAI(api_key='',
                timeout=10.0, max_retries=5)

# SET OPENAI API KEY

MODEL = 'gpt-3.5-turbo'

SOURCE = 'music'
TARGET = 'book'
METHOD = 'top5'

def completePrompt(p, model, instruction):
    response = client.completions.create(model=model,
    prompt = instruction + p + "\n\n",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return(response.choices[0].text)

def doQuery(p, model, instruction, ans):
    sysout = completePrompt(p, model, instruction)
    sysout = sysout.strip()
    print(p, "System:", sysout)
    sysout = sysout[:len(ans)]
    return(sysout == ans)

def chatCompletPrompt(p, model):
    #try:
    response = client.chat.completions.create(model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p}
        ],
        max_tokens=5,
    )
    #except TypeError:
    #    print()
    #print(response.choices[0].message)
    return response.choices[0].message.content

def doChatQuery(p, model, ans):
    time.sleep(0.1)
    sysout = chatCompletPrompt(p, model)
    sysout = sysout.strip()
    #print(p, "System:", sysout)
    try:
        sysout = re.search(r'Label:\s?(\w+)', sysout, re.I).groups()[0]
    except TypeError:
        sysout = 'x'
        print('Answer Not Found')
    except AttributeError:
        pass
    return(sysout == ans)

# results File: runID - # dd/mm/YY H:M:S
dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")

# CREATE RESULTS FILE
resFile = Path("runs") / dt_string
resFile.touch(exist_ok=False)
resFile.write_text(f"{SOURCE}2{TARGET}, {METHOD}")

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

    record = []

    # RUN THROUGH EXAMPLE FILES
        
    with open(f'./data/{METHOD}_50/{SOURCE}2{TARGET}_{METHOD}_filter50.json', 'r') as source:
        examples = json.load(source)

        print(f"Source: {SOURCE}, Target: {TARGET}")

        correct = 0
        # RUN THROUGH EXAMPLES
        pbar = tqdm(total=len(examples))
        for i, e in enumerate(examples):
            prefix = f"Given the input sentence, assign a sentiment label from ['positive', 'neutral', 'negative']. Give your response with the answer label only. Do not include irrelevant text. \n"
            demo_text = ''
            for demo in e['source']:
                demo_text += f'Sentence: {demo["s_sent"]}Label: {label2text(demo["s_label"])}\n'
            prompt = prefix + demo_text + f"\nSentence: {e['q_sent']}Label:"
            answer = label2text(e['q_label'])
            #res = doQuery(prompt, ARGS.model, instructions, answer)
            res = doChatQuery(prompt, MODEL, answer)

            if res is True:
                correct += 1
            pbar.update()
            pbar.set_postfix_str(f'Acc: {correct/(i+1)}')
            
            record.append({i:res})
        with resFile.open("a") as f:
            f.write(f'Accuracy: {correct/len(examples)}')
    print(f"iteration {iteration}")
