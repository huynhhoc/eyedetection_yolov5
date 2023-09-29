import cv2
import numpy as np
import streamlit as st
from detect_update import detect
import numpy as np
import os
from PIL import Image
import logging
import sys
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

def load_detect_from_yolov5():
    path_files = detect(source="runs/streamlits", weights='weights/yolov5_eyedetv3.pt', data='data/eye.yaml', save_crop=True)
    return path_files
#------------------------------------------------------------------------------------------------
def uploadImage() -> None:
    path = os.getcwd()
    for img in os.listdir(path + '/runs/streamlits'):
        print ("img: ", img)
        os.remove(path + '/runs/streamlits/'+img)
    uploaded_file = st.sidebar.file_uploader("Choose a image file", type=['jpg', 'jpeg', 'png'])
    path = os.getcwd()
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.sidebar.image(image, channels="BGR")
        img_array = np.array(image)
        cv2.imwrite(f'runs/streamlits/{uploaded_file.name}',img_array)
        #source = f'runs/streamlits/{uploaded_file.name}'
    
        if st.button('Detect'):
            path_files = load_detect_from_yolov5()
            print ("path_files: ", path_files)
            col1, col2 = st.columns(2)
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
                            st.image(path + '/'+filename, channels="BGR")
                    else:
                        st.text(f'Sorry, I can not detect any eyes in {uploaded_file.name}')
                        print("Cannot detect eye: " + uploaded_file.name + ", please doulbe check that image in the folder" + path_files[0])
                        log.info("Cannot detect eye: " + uploaded_file.name + ", please doulbe check that image in the folder" + path_files[0])
            except Exception as ex:
                print(ex)
                log.info(ex)
#----------------------------------------------------------------------------
if __name__ == '__main__':
    #log = createLogger()
    try:
        name= request.remote_addr
        print("ip___________: ", name)
    except:
        name = "Hoc"
    head = st.container()
    upload_sec = st.container()
    with head:
        st.title("Eye Detection - by Torus Actions")
    #-------------------------------
    with upload_sec:
        try:
            source = uploadImage()
        except Exception as ex:
            print(ex)
            log.info(ex)
    #-----------------------------