EYE DETECTION:
    Yolov5 has been modified to detect eye object, below are modified contains:
        eye.yaml:
            # Classes
            nc: 11 # number of classes
            names: ['0','1','2','3','4','5','6','7','8','9','10']  # class names
            Explanation: although we just detect left eye and right eye, several images from the training dataset have more than one people, it means that those pictures have more than 2 eyes. Elevent is the maximum number of eyes that exist in some pictures in the training dataset.
        detect_update.py:
            detect_update.py is a copy version of detect.py with only one updated: the boundary of croped images are extended, we will try to modify to suitable requirements.
            new lines: 160 - 162
        weights:
            weights/yolov5_eyedetv3.pt
            The weight was buit based on 7157 images
    
    Training:
        images  - 7157 images
        labels  - 7157 yolo_lables
    Validation:
        images  - 481 images
        labels  - 481 labels
