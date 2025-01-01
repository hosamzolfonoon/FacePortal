import cv2
import mediapipe as mp
from datetime import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import SecondLocator
import seaborn as sns
import warnings
import joblib
import tensorflow as tf
import base64
from flask import Flask, request, jsonify

class FaceMechDetector():
    def __init__(self, refine_landmarks=True,
                 staticMode=True, max_number_face=1,
                 min_detection_confidence=0.6,
                 min_tracking_confidence=0.6):
        self.staticMode = staticMode
        self.max_number_face = max_number_face
        self.refine_landmarks = refine_landmarks
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,
                                                 self.max_number_face,
                                                 self.refine_landmarks,
                                                 self.min_tracking_confidence,
                                                 self.min_detection_confidence)

    def findFaceMech (self, img, draw=True):
        face_landmarks_dict = {}

        if draw :
            self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceMesh.process(self.imgRGB)

            if self.results.multi_face_landmarks:
                for faceLms in self.results.multi_face_landmarks:
                    height, width, channel = img.shape # it very critical point: it is equal to y, x, z not x, y, z
                    for id, lm in enumerate(faceLms.landmark):
                        #x, y, z = int(lm.x*width), int(lm.y*height), int(lm.z*channel)
                        x, y = int(lm.x*width), int(lm.y*height)
                        #face_landmarks_dict[id] = [x, y, z]
                        face_landmarks_dict[id] = [x, y]
        return face_landmarks_dict



detector_face = FaceMechDetector(staticMode=False)



def x_img_face_prev ():
    column_list_items = columns_list_face()
    empty_array = np.zeros((1, len(column_list_items)-1)) # 'column_list' function includes 'position' which should be remove in features extraxtion
    X_img_face_prev = pd.DataFrame(empty_array, columns=columns_list_face()[:-1]) # 'column_list' function includes 'position' which should be remove in features extraxtion
    return X_img_face_prev


def detection_list_function_face():
    max_min_list = [234, 454, 10, 5] # Left, Right, Top, Middle
    outer_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    left_eye = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
    left_eyebrow = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
    right_eye = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
    right_eyebrow = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    ## Merging all face necessary landmarks
    b = max_min_list+outer_lip+left_eye+left_eyebrow+right_eye+right_eyebrow#
    b.sort()
    detection_list = []
    [detection_list.append(i) for i in b if i not in detection_list] # it has 184 landmarks
    return detection_list


def emotion_folder_list_function():
    emotion_folder_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Normal', 'Sad', 'Surprise'] # S06
    return emotion_folder_list


def columns_list_face():
    detection_list = detection_list_function_face()
    column_list_items = []
    for i in detection_list:
        column_list_items.append(str(i) + '_X')
        column_list_items.append(str(i) + '_Y')
    column_list_items.append('emotion')
    return column_list_items


def norm_list(raw_list):
    if min(raw_list) == max(raw_list):
            normmalzied_list = [0]*len(raw_list)
    else:
        normmalzied_list = [round((float(i)-min(raw_list))/(max(raw_list)-min(raw_list)),3) for i in raw_list]
    return normmalzied_list


def norm_line_list(raw_list, max_y):
    # ** a.X + b = Y <<<<< calculate nanmarks Y below of landmark 5-Y  based on line equation
    a = 1/(max_y-min(raw_list)) # a = (Xmax - Xmin)/(Ymax-Ymin) <<<<< Xmax=1 & Xmin=0 because of normalization Ys between landmark 10 and 5
    b = -1*min(raw_list)/(max_y-min(raw_list)) # b = -Ymin/(Ymax-Ymin)
    if min(raw_list) == max(raw_list):
        normmalzied_line_list = [0]*len(raw_list)
    else:
        normmalzied_line_list = [round((a*float(i)+b),3) for i in raw_list]
    return normmalzied_line_list


def predict_df_generator_face(img):
    prediction_message = ''
    faceLandmrksDict = detector_face.findFaceMech(img)
    column_list_items = columns_list_face()
    empty_array = np.zeros((1, len(column_list_items)-1)) # 'column_list' function includes 'emotion' which should be remove in features extraxtion
    items = detection_list_function_face()
    j = 0
    if len(faceLandmrksDict) == 478:
        raw_list_x = []
        raw_list_y = []
        max_y = faceLandmrksDict[1][1]
        for item in items:
            raw_list_x.append(faceLandmrksDict[item][0])
            raw_list_y.append(faceLandmrksDict[item][1])
        normalized_raw_list_x = norm_list(raw_list_x)
        normalized_raw_list_y = norm_line_list(raw_list_y, max_y)
        for i in range(len(items)):
            empty_array[j, i*2] = normalized_raw_list_x[i]
            empty_array[j, i*2+1] = normalized_raw_list_y[i]
    else:
        prediction_message = 'No Face'
    X_img_face = pd.DataFrame(empty_array, columns=columns_list_face()[:-1]) # 'column_list' function includes 'emotion' which should be remove in features extraxtion
    return X_img_face, prediction_message


def prediction_face(model_mlp, model_svm, model_rf, model_dt, X_img_face, prediction_message):
    prediction_message_list = []
    if prediction_message != 'No Face':
        emotion_folder_list = emotion_folder_list_function()
        pred_mlp = int(model_mlp.predict(X_img_face))
        pred_svm = int(model_svm.predict(X_img_face))
        pred_rf = int(model_rf.predict(X_img_face))
        pred_dt = int(model_dt.predict(X_img_face))
        prediction_message_list.append('*** FER ***' )
        prediction_message_list.append('MLP : '+str(emotion_folder_list[pred_mlp]))
        prediction_message_list.append('SVM : '+str(emotion_folder_list[pred_svm]))
        prediction_message_list.append('RF  : '+str(emotion_folder_list[pred_rf]))
        prediction_message_list.append('DT  : '+str(emotion_folder_list[pred_dt]))
    else:
        prediction_message_list.append('*** FER ***' )
        prediction_message_list.append(prediction_message)
    return prediction_message_list

def models_running_face():
    ### I you want to check especiasl emoptions consider emotion_folder_list ###
    model_dnn = tf.keras.models.load_model('/ferapp/FER_DNN.keras')
    model_mlp = joblib.load('/ferapp/MLP_Face.pkl') # Loading trained model # Consider the addrerss of file
    model_svm = joblib.load('/ferapp/SVM_Face.pkl') # Loading trained model # Consider the addrerss of file
    model_rf = joblib.load('/ferapp/RF_Face.pkl') # Loading trained model # Considering training time # Consider the addrerss of file
    model_dt = joblib.load('/ferapp/DT_Face.pkl')
    return model_dnn, model_mlp, model_svm, model_rf, model_dt

def put_text_face(img, predict_message_final, width_in, height_in):
    font_scale =.7 #  round(0.0009 * width_in)
    height = round(height_in * 0.8)
    height_line_space = round(height_in * 0.2)
    for i in range(len(predict_message_final)):
        cv2.putText(img, predict_message_final[i], (round(0.2*width_in), height), cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=(0,0,0), thickness=1)
        height += round(height_line_space/5)

def predict_message_final_face_df_function (face_df_array, predict_message_final_face):
    global ml_list
    face_df_array_hlep = np.empty((1, len(ml_list)), dtype='object')
    if len(predict_message_final_face) == 2:
        for counter_face_df in range(len(ml_list)):
            face_df_array_hlep[0,counter_face_df] = np.nan
    elif len(predict_message_final_face) == 5:
        predict_message_final_face = predict_message_final_face[1:]
        for counter_face_df in range(len(ml_list)):
            emotion_help = predict_message_final_face[counter_face_df][6:]
            face_df_array_hlep[0,counter_face_df] = emotion_help
    face_df_array = np.concatenate((face_df_array, face_df_array_hlep))
    return face_df_array

def prediction_dnn(model, X_img, prediction_message):
    prediction_message_list = []
    if prediction_message != 'No Face':
        emotion_folder_list_items = emotion_folder_list_function()
        # Predict on coords_small_test
        preds = model.predict(X_img) # the output is only one row
        array_help = preds[0]
        #preds_rounded = np.round(array_help, 4)
        pred_list = list(array_help)
        pred_list = [round(i*100,2) for i in pred_list]
        preds_chosen = np.argmax(pred_list)
        prediction_message_list.append('***  >>>  Dominante Prediction : '+str(emotion_folder_list_items[preds_chosen])+'  <<<  ***')
        #prediction_message_list.append('==============================')
        for i in range(len(emotion_folder_list_items)):
            prediction_message_list.append('Label "'+str(emotion_folder_list_items[i])+'" = %'+str(pred_list[i]))
    else:
        prediction_message_list.append('***  FER DNN  ***' )
        prediction_message_list.append(prediction_message)
    return prediction_message_list

def predict_message_final_face_df_function_dnn (face_df_array, predict_message_list_dnn):
    global ml_list_dnn
    face_df_array_hlep = np.empty((1, len(ml_list_dnn)), dtype='object')
    if len(predict_message_list_dnn) == 2:
        for counter_face_df in range(len(ml_list_dnn)):
            face_df_array_hlep[0,counter_face_df] = np.nan
    elif len(predict_message_list_dnn) == 8:
        emotion_help = predict_message_list_dnn[0][33:-10]
        face_df_array_hlep[0,0] = emotion_help
    face_df_array = np.concatenate((face_df_array, face_df_array_hlep))
    return face_df_array


def put_text_face_dnn(img, predict_message_list, width_in, height_in):
    font_scale =.7 #  round(0.0009 * width_in)
    height = round(height_in * 0.8)
    height_line_space = round(height_in * 0.2)
    for i in range(len(predict_message_list)):
        if i < 5:
            cv2.putText(img, predict_message_list[i], (round(0.35*width_in), height+i*round(height_line_space/5)), cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=(0,0,0), thickness=1)
        else:
            j = i-4
            cv2.putText(img, predict_message_list[i], (round(0.58*width_in), height+j*round(height_line_space/5)), cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=(0,0,0), thickness=1)

prev_landmarks = None
prev_directions = None
prev_time = None
face_df_array = None
face_df_array_dnn = None
ml_list = ['MLP', 'SVM', 'RF', 'DT']
ml_list_dnn = ['DNN']
check = False
color_list = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan'] # Definition color lists
duration_time = 0
pTime = 0 # Auxiliary variable, # Hyperparameter to control FPS
face_detection_counter = 0
model_dnn, model_mlp_face, model_svm_face, model_rf_face, model_dt_face = models_running_face()
face_df_array = np.empty((1, len(ml_list)), dtype='object')
face_df_array_dnn = np.empty((1, len(ml_list_dnn)), dtype='object')

app = Flask(__name__)

@app.route('/', methods=['POST'])
def upload():
    global model_dnn, model_mlp_face, model_svm_face, model_rf_face, model_dt_face
    global model_mlp_body, model_svm_body, model_rf_body, model_dt_body
    global face_df_array, face_df_array_dnn, body_df_array
    global ml_list, ml_list_dnn
    global prev_landmarks, prev_directions, prev_time, duration_time
    global df_help_1, df_help_2, columns_list
    global check, color_list, pTime, face_detection_counter

    try:
        # Parse JSON data
        data = request.json
        required_fields = ['image', 'timestamp', 'process_duration', 'frame_rate', 'detection_threshold', 'dimensions']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Invalid request format"}), 400
        process_duration = data['process_duration']

        if duration_time < process_duration:
            # Extract parameters
            frame_rate = data['frame_rate']
            dimensions = data['dimensions']
            width_in = dimensions['width']
            height_in = dimensions['height']
            cTime = time.time()
            time_elapsed = cTime - pTime # Necessary variable to control the frame rate
                    # Decode the base64 image
            try:
                image_data = base64.b64decode(data['image'].split(",")[1])  # Remove 'data:image/jpeg;base64,' prefix
            except (base64.binascii.Error, IndexError) as e:
                return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400
            # Decode the base64 image
            np_arr = np.frombuffer(image_data, np.uint8)
            
            try:
                np_arr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Decoded image is None, possibly invalid image format.")
            except Exception as e:
                return jsonify({"error": f"Error decoding image: {str(e)}"}), 400
            
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if time_elapsed > 1.0/frame_rate:
                ## From Face Recognition
                X_img_face, prediction_message_face = predict_df_generator_face(img)
                predict_message_final_face = prediction_face(model_mlp_face, model_svm_face, model_rf_face, model_dt_face, X_img_face, prediction_message_face)
                predict_message_list_dnn = prediction_dnn(model_dnn, X_img_face, prediction_message_face)
                #face_df_array = predict_message_final_face_df_function (face_df_array, predict_message_final_face)
                #face_df_array_dnn = predict_message_final_face_df_function_dnn (face_df_array_dnn, predict_message_list_dnn)
                put_text_face(img, predict_message_final_face, width_in, height_in)
                put_text_face_dnn(img, predict_message_list_dnn, width_in, height_in)

                return jsonify({
                    "predict_message_final_face": predict_message_final_face,
                    "predict_message_list_dnn": predict_message_list_dnn,
                })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0")