import cv2
from cv2 import threshold
import numpy as np
import streamlit as st
from detect_update import detect
import numpy as np
import os
from PIL import Image
import logging
import sys
import socket
import shutil

from flask import request

log = logging.getLogger()
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s','%m-%d-%Y %H:%M:%S')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs/logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

log.addHandler(file_handler)
log.addHandler(stdout_handler)

def get_all_files_subfolders(folder: str) -> list:
    listOfFiles = list()
    files = list()
    for (dirpath, _, filenames) in os.walk(folder):
        for file in filenames:
            if file not in files:
                files.append(file)
                listOfFiles.append(os.path.join(dirpath, file))
    return listOfFiles

def load_detect_from_yolov5(namefolder, thres_hold):
    path_files = detect(source="runs/streamlits/" + namefolder, weights='weights/yolov5_8709.pt', data='data/eye.yaml', conf_thres = thres_hold,save_crop=True, device='cpu')
    return path_files
#------------------------------------------------------------------------------------------------
def uploadImage(head, upload_sec) -> None:
    path = os.getcwd()
    try:
        name= request.remote_addr
        log.info("client: " + name)
    except:
        name = "localhost"
    name = socket.gethostname()
    if not os.path.exists(path + '/runs/streamlits/'+ name):
        os.mkdir(path + '/runs/streamlits/'+ name)
    for img in os.listdir(path + '/runs/streamlits/'+ name +"/"):
        os.remove(path + '/runs/streamlits/' + name + "/"+img)
    uploaded_file = st.sidebar.file_uploader("Choose a image file", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.sidebar.image(image, channels="BGR")
        img_array = np.array(image)
        cv2.imwrite(f'runs/streamlits/{name}/{uploaded_file.name}',img_array)
        col1, col2 = st.columns(2)
        try:
            thres_hold = 20
            with head:
                thres_hold = st.slider("Confidence Threshold",0,100, 20)
            with upload_sec:
                with st.spinner("Detecting ...."):
                    if st.button('Detect'):
                        path_files = load_detect_from_yolov5(name,(thres_hold-1)/100.0)
                        print ("path_files: ", path_files)
                        with col1:
                            st.header("Results:")
                            for filename in path_files:
                                st.image(path + '/'+filename, channels="BGR")
                        # extract eyes
                        try:
                            index_last_slash = path_files[0].rfind("/")
                            sub_folder = path_files[0][: index_last_slash]
                            sub_folder = sub_folder +"/crops"
                            lstimages = get_all_files_subfolders(sub_folder)
                            lstimages = list(dict.fromkeys(lstimages))
                            with col2:
                                st.header("Eyes Extraction:")
                                nimages = len(lstimages)
                                if nimages > 0:
                                    for filename in lstimages:
                                        label = filename.split("_")[-1][2:-4]
                                        st.image(path + '/' + filename, channels="BGR")
                                        st.text("Confidence threshold: " + label)
                                else:
                                    st.text(f'Sorry, I can not detect eye')
                                    print("Client:" + name + ". Cannot detect eye: " + uploaded_file.name + ", please doulbe check that image in the folder " + path_files[0])
                                    log.info("Client:" + name + ". Cannot detect eye: " + uploaded_file.name + ", please doulbe check that image in the folder " + path_files[0])
                                    if not os.path.exists(path + '/runs/undetect'):
                                        os.mkdir(path + '/runs/undetect')
                                    shutil.copy(path + "/"+ path_files[0], path + '/runs/undetect')
                        except Exception as ex:
                            print("Client:" + name + ", " + ex)
                            log.info("Client:" + name + ", " + ex)
        except Exception as ex:
            print ("error: ", ex)
            st.text("Server is busy, please try again later!")

#----------------------------------------------------------------------------
if __name__ == '__main__':

    head = st.container()
    upload_sec = st.container()
    with head:
        st.title("Eye Detection - by Torus Actions")
    
    uploadImage(head,upload_sec)