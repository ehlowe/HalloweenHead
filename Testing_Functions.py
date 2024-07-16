import os
import time
import base64
import cv2
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError

with open("working_files/SECRET.txt","r") as f:
    OPENAI_KEY=f.readline().split("\n")[0]


os.environ["OPENAI_API_KEY"] = OPENAI_KEY
client = OpenAI()



def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def get_openai_response(the_prompt):
    client = OpenAI()
    image_path="working_files/vision_description_image.jpeg"
    while True:
        im=cv2.imread("working_files/shi.jpg")
        initial_size = os.path.getsize("working_files/shi.jpg")
        time.sleep(0.01)
        if (os.path.getsize("working_files/shi.jpg") == initial_size) and (initial_size>300):
            cv2.imwrite(image_path,im)
            break

    base64_image = encode_image(image_path)
    image_url=f"data:image/jpeg;base64,{base64_image}"

    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {'role': 'user', 'content': [{"type": "text", "text":the_prompt},{"type": "image_url", "image_url": image_url},],}
            ],
            temperature=0.4,
            max_tokens=500,
            stream=True
        )
        return response
    except Exception as e:
        print(f"An exception occurred: {e}")
        return None
    

the_prompt="I want you to role play as a talking skeleton head who is whitty and funny. You are greeting trick-or-treaters on Halloween night at the front door of a house. I want this conversation to be engauging for the trick or treaters and for it to be funny. Try to be a little crazy and try not to repeat yourself or say things that are repetitive and don't ramble on for a long time because this is a back and forth conversation.\nYou: <message>\n\nWhat should you say for the <message> to the user?\nAnswer: "
print("PROMPT: ",the_prompt)
timeout_seconds=2

response=None
while response==None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_openai_response, the_prompt)
        try:
            response = future.result(timeout=timeout_seconds)
        except TimeoutError:
            print(f"Timeout after {timeout_seconds} seconds")
            response = None

print("MADE RESPONSE")
collected_messages=[]
for chunk in response:
    try:
        print(chunk)
        #shared_dict["last_token_size"]=len(chunk['choices'][0]['delta']["content"])
        print(chunk.choices[0].content)
        collected_messages.append(chunk.choices[0].delta.content)
        #shared_dict["text"]="".join(collected_messages)
    except Exception as e:
        print(e)
result_s="".join(collected_messages)