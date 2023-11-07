import openai
import sys
import os

# Read all keys
with open("working_files/SECRET.txt",'r') as f:
    list_of_keys=f.read().split("\n")
    OPENAI_SECRET=list_of_keys[0]
    TOGETHER_SECRET=list_of_keys[1]
    REPLICATE_SECRET=list_of_keys[2]

# Setup OpenAI for transcription
os.environ['OPENAI_API_KEY'] = OPENAI_SECRET
openai.api_key=OPENAI_SECRET
import time

t_start=time.time()
response = openai.ChatCompletion.create(
    #model='gpt-3.5-turbo',
    model="gpt-4",
    messages=[
        {'role': 'user', 'content': "Hi can you tell me about world history!"}
    ],
    temperature=0,
    max_tokens=500,
    stream=True  # this time, we set stream=True
)

collected_messages=[]
for chunk in response:
    print(time.time()-t_start)
    try:
        print(chunk['choices'][0]['delta']["content"])
        collected_messages.append(chunk['choices'][0]['delta']["content"])
    except Exception as e:
        print(e)
    print("".join(collected_messages))