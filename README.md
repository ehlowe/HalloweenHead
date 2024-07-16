# HalloweenHeadPrivate

The AITracking.py file communicates with the arduino and does the AI tracking while the CoversationalBehavioral.py file controls the AI chat, mouth, and eyes. The head_control_arduino.ino is the arduino code which connects to the laptop serially and controls the head.

Here is a [youtube video showing it in action](https://www.youtube.com/watch?v=PIQqx56Qymg).


![image](https://github.com/user-attachments/assets/8b0f52de-12e0-4397-a366-750fe0afec45)

## Explanation:
Inside the head is a camera which serves as the method to track trick or treaters, talk about their outfit, and record audio for transcription purposes. This data stream goes to my laptop through cables that run under the door. The laptop runs the human head bounding box inference and handles all of the controls logic. The eye color, mouth state, and head position write information is sent from the laptop to the arduino to control the head. 

When a trick or treater approaches the horizontal rotation of the head changes to point toward the door which puts the robot in the approaching state, once they reach the door the head position triggers talking state and the chatbot says something to the trick or treaters. Once the chatbot has said something the microphone enables again, if a certain noise level is reached for a certain period it waits until the noise level drops off for a period to indicate that someone spoke. This was an unusually hard problem to solve because the sound of the servos I wanted to use were louder than a person speaking from the camera's (microphone's) perspective. I use small servos with mechanical leverage to get the head to turn quietly.

After speech is detected, the request is made for the chatbot to start speaking. The speaking process uses live streamed tokens from the LLM and parses the tokens such that gTTS (google text to speech) can create a continuous stream of speech segments as the response is generated. This allows for fast response times (2-3 seconds) when the generated audio files are played as they are ready. 

Once the trick or treaters start walking away the states and conversation are reset allowing for new trick or treaters to come up and interect with the bot.

