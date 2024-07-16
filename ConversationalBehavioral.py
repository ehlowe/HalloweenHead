# Motion
import serial
import time
import multiprocessing
import socket

# Conversation
import threading
import os
import openai
from openai import OpenAI
import time
from pvrecorder import PvRecorder
import wave
import struct
import numpy as np


# Import the functions for conversation
import customlibrary.HalloweenHead as hh


# Socket connection
def socket_connection(shared_dict):
    import socket
    import select

    print(socket.gethostname())

    hostname = socket.gethostname()
    port=1234
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    #set timeout
    s.settimeout(2)

    s.bind((hostname, port))
    s.listen(1)
    try:
        conn_obj, addr = s.accept()
    except:
        pass

    fail_check_time=time.time()
    failed_check_times=0


    while True:
        try:
            data=conn_obj.recv(1024)
            data_string=data.decode()
            if data_string!="":
                numbers_l=data_string.split(",")
                shared_dict["x_pos"]=int(numbers_l[0])
                shared_dict["y_pos"]=int(numbers_l[1])
                if int(numbers_l[2])==1:
                    shared_dict["failed"]=True
                else:
                    shared_dict["failed"]=False

            mouth=shared_dict["mouth"]
            eye=shared_dict["eye"]
            send_message=f"{str(mouth)},{str(eye)}\n"
            conn_obj.sendall(send_message.encode())

            # Update the approaching and at_door variables
            if (time.time()-fail_check_time)>0.2:
                if not shared_dict["failed"]:
                    failed_check_times=0
                    if shared_dict["x_pos"]>1500:
                        shared_dict["approaching"]=True
                        shared_dict["at_door"]=False
                    else:
                        shared_dict["approaching"]=False
                        shared_dict["at_door"]=True
                else:
                    failed_check_times+=1

                if (failed_check_times)>10:
                    shared_dict["approaching"]=False
                    shared_dict["at_door"]=False
                
                fail_check_time=time.time()


            
        except Exception as e:
            print("read or send error: ", e, " line: ", e.__traceback__.tb_lineno)
            try:
                conn_obj, addr = s.accept()
            except:
                print("Tried to reconnect, but failed")


# Vision Description
def description_task(shared_dict):
    import cv2
    while True:
        im=cv2.imread("working_files/shi.jpg")
        initial_size = os.path.getsize("working_files/shi.jpg")
        time.sleep(0.01)
        if (os.path.getsize("working_files/shi.jpg") == initial_size) and (initial_size>300):
            cv2.imwrite("working_files/vision_description_image.jpg",im)
            break

    vision_output = replicate.run(
        "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
        input={"image": open("working_files/vision_description_image.jpg", "rb"),"prompt":"In this image is: "}
    )
    description_string=""
    for item in vision_output:
        description_string+=item
        shared_dict["visual_description"]=description_string


    


if __name__ == "__main__":
    if not os.path.isdir("working_files/audio_files"):
        os.mkdir("working_files/audio_files")
    if not os.path.isdir("working_files/tts_files"):
        os.mkdir("working_files/tts_files")

    #create thread for motion
    import threading
    import keyboard
    import cv2

    # Read all keys
    with open("working_files/SECRET.txt",'r') as f:
        list_of_keys=f.read().split("\n")
        OPENAI_SECRET=list_of_keys[0]
        TOGETHER_SECRET=list_of_keys[1]
        REPLICATE_SECRET=list_of_keys[2]
    
    # Setup OpenAI for transcription
    os.environ['OPENAI_API_KEY'] = OPENAI_SECRET
    openai.api_key=OPENAI_SECRET
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    # Setup together LLM
    model_name="togethercomputer/llama-2-13b-chat"
    together_model_name="togethercomputer/llama-2-70b-chat"
    import together
    together.api_key = TOGETHER_SECRET

    # Setup replicate Vision LLM
    import replicate
    os.environ["REPLICATE_API_TOKEN"]=REPLICATE_SECRET

    # Setup shared data 
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    #Audio
    shared_dict["audio"]=[]
    shared_dict["savebool"]=True
    shared_dict["saveduration"]=0
    shared_dict["playingsound"]=True

    #TTs
    shared_dict["text"]=""
    shared_dict["file queue"]=[]
    shared_dict["last_token_size"]=0
    shared_dict["llm_finished"]=False
    shared_dict["tts_finished"]=False

    # Animation
    shared_dict["mouth"]=0
    shared_dict["eye"]=1

    # Vision
    shared_dict["visual_description"]=""

    # Motion
    shared_dict["x_pos"]=None
    shared_dict["y_pos"]=None
    shared_dict["failed"]=True
    shared_dict["approaching"]=False
    shared_dict["at_door"]=False



    socket_thread = threading.Thread(target=socket_connection, args=(shared_dict,))
    socket_thread.start()

    being_smart=True

    #start threads
    recording_thread = threading.Thread(target=hh.recorder, args=(shared_dict,))
    recording_thread.start()

    #sound thread
    sound_thread=threading.Thread(target=hh.tts_player, args=(shared_dict,))
    sound_thread.start()

    #gTTS thread
    tts_thread=threading.Thread(target=hh.tts_generator,args=(shared_dict,))
    tts_thread.start()



    chat_log=""

    path_i=0
    #while recording thread is running, do this
    approaching_to_door=False
    prev_approaching_to_door=False
    approach_ready=False

    t_print=time.time()
    off_bool=False

    t_failed_start=time.time()

    ran_vision_description=False

    openai_bool=False
    together_bool=True
    if openai_bool:
        together_bool=False
    if together_bool:
        openai_bool=False

    import base64
    import os



    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_openai_response(the_prompt):
        client = OpenAI()
        image_path="working_files/vision_description_image.jpg"
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
                    {'role': 'user', 'content': [{"type": "text", "text":the_prompt},{"type": "image", "image": image_url},],}
                ],
                temperature=0.4,
                max_tokens=500,
                stream=True
            )
            return response
        except Exception as e:
            print(f"An exception occurred: {e}")
            return None
    

    while recording_thread.is_alive():
        time.sleep(0.05)

        if shared_dict["approaching"]:
            approaching_to_door=True

        if not ran_vision_description:
            if (time.time()-t_failed_start)>0.4:
                description_thread=threading.Thread(target=description_task, args=(shared_dict,))
                #description_thread.start()
                ran_vision_description=True
        
        prev_approaching_to_door=approaching_to_door

        if approaching_to_door:
            if shared_dict["at_door"]:
                approach_ready=True
                approaching_to_door=False

        if shared_dict["failed"]:
            if (time.time()-t_failed_start)>2:
                approach_ready=False
                shared_dict["savebool"]=True
                chat_log=""
                t_failed_start=time.time()
                ran_vision_description=False
        else:
            t_failed_start=time.time()

        if keyboard.is_pressed('r'):
            approach_ready=True
            print("\nAPPROACH READY ON\n")
        # if keyboard.is_pressed('c'):
        #     off_bool=True
        # if keyboard.is_pressed('v'):
        #     off_bool=False

        if time.time()-t_print>1:
            print("Failed: ", shared_dict['failed'], "Approaching: ", shared_dict["approaching"], "At Door: ", shared_dict["at_door"], "Approach Ready: ", approach_ready," Approaching to door: ", approaching_to_door)
            t_print=time.time()

        # if off_bool:
        #     shared_dict["text"]=""
        #     shared_dict["savebool"]=False

        #if recording thread has saved audio, do this
        if shared_dict["savebool"] and approach_ready and not off_bool:
            if not ran_vision_description:
                description_thread=threading.Thread(target=description_task, args=(shared_dict,))
                #description_thread.start()
                ran_vision_description=True
            shared_dict["playingsound"]=True
            path_i+=1
            shared_dict["savebool"]=False
            hh.save_audio(shared_dict["audio"],shared_dict["saveduration"],"working_files/audio_files/"+str(path_i)+".wav")
            print(shared_dict["saveduration"]," Saving audio ", path_i)

            #stop listening
            t_thinking_start=time.time()


            if being_smart:
                #transcribe
                t_start=time.time()
                if chat_log!="":
                    shared_dict["eye"]=2
                    audio_file= open("working_files/audio_files/"+str(path_i)+".wav", "rb")
                    try:
                        transcript = openai.Audio.transcribe("whisper-1", audio_file)
                    except:
                        transcript="You couldn't hear me"
                        break
                else:
                    shared_dict["eye"]=2
                    transcript={"text":""}
                t_end=time.time()
                print("Time to transcribe: ",t_end-t_start, " Text:",transcript["text"])

                ts=time.time()
                # generate response
                if chat_log=="":
                    the_prompt="I want you to role play as a talking skeleton head who is whitty and funny. You are greeting trick-or-treaters on Halloween night at the front door of a house. I want this conversation to be engauging for the trick or treaters and for it to be funny. Try to be a little crazy and try not to repeat yourself or say things that are repetitive and don't ramble on for a long time because this is a back and forth conversation. This is what you see in front of you: "+shared_dict["visual_description"]
                    chat_log=the_prompt
                    the_prompt+="\nYou: <message>\n\nWhat should you say for the <message> to the user?\nAnswer: "
                else:
                    chat_log=chat_log+"\nUser: "+str(transcript["text"])
                    the_prompt=chat_log+"\nYou: <message>\n\nWhat should you say for the <message> to the user?\nAnswer: "
                
                #open a yaml file to write the prompt to the end
                with open("working_files/prompt.yaml","a") as f:
                    #get rid of emojis from prompt
                    write_prompt=the_prompt.encode('ascii', 'ignore').decode('ascii')
                    f.write(write_prompt+"\n\n\n\n\n")

                print("the prompt test: ",the_prompt)
                #the_prompt="I've seen more candy in a single bag! But, hey"
                ts=time.time()
                # generate response
                result_l=[]
                shared_dict["file queue"]=[]
                shared_dict["time_started"]=time.time()
                result_s=""
                shared_dict["llm_finished"]=False
                shared_dict["tts_finished"]=False
                shared_dict["text"]=""

                timeout_seconds = 2
                response=None
                if openai_bool:
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
                            shared_dict["last_token_size"]=len(chunk['choices'][0]['delta']["content"])
                            print(chunk['choices'][0]['delta']["content"])
                            collected_messages.append(chunk['choices'][0]['delta']["content"])
                            shared_dict["text"]="".join(collected_messages)
                        except Exception as e:
                            print(e)
                    result_s="".join(collected_messages)


                if together_bool:
                    # Together llamma response:
                    for token in together.Complete.create_streaming(prompt=the_prompt,model=together_model_name,max_tokens=500,temperature=0.9):
                        print(token, end="", flush=True)
                        shared_dict["last_token_size"]=len(token)
                        result_l.append(token)
                        result_s="".join(result_l).strip()
                        shared_dict["text"]=result_s
                print("LLM RESULT: ",result_s)

                shared_dict["llm_finished"]=True
                chat_log=chat_log+"\nYou: "+result_s.split("You: ")[-1].split("</s>")[0]
                with open("working_files/prompt.yaml","a") as f:
                    #get rid of emojis from prompt
                    write_prompt=chat_log.encode('ascii', 'ignore').decode('ascii')
                    f.write(write_prompt+"\n\n\n\n\n")
                print("\n")
                print("LLM Time: ",time.time()-ts)

