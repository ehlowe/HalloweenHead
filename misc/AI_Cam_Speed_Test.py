import cv2



def cam_function(shared_dict):
    cam=cv2.VideoCapture(1, cv2.CAP_MSMF)

    t_start=time.time()
    while True:
        try:
            t_start=time.time()
            ret,frame=cam.read()
            shared_dict["im"]=frame
            time.sleep(0.024)
            print("CAM: ",1/(time.time()-t_start))
        except Exception as e:
            print(e)
            time.sleep(0.024)

if __name__=="__main__":
    import torch
    import os
    import time
    import threading
    import multiprocessing as mp

    manager = mp.Manager()
    shared_dict = manager.dict()
    shared_dict["im"]=None

    cam_thread=threading.Thread(target=cam_function, args=(shared_dict,))
    cam_thread.start()

    model = torch.hub.load("ultralytics/yolov5", 'custom', path="working_files/crowdhuman_yolov5m.pt")

    prev_frame_file_size=0


    while True:
        t_start=time.time()
        #if the image file is the same size as the previous iteration run inference
        frame=shared_dict["im"]
        try:
            results = model(frame)
            cv2.imshow("frame",frame)
            cv2.waitKey(1)
        except Exception as e:
            print(e)

        try:
            print(1/(time.time()-t_start))
        except:
            pass
