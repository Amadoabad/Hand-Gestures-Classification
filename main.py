import joblib
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2
import statistics as st



model = joblib.load('exports/hand_gesture_final_model.pkl')
encoder = joblib.load('exports/encoder.pkl')

def get_tst_points(landmark, order = False):
    """Converts the output of hand_landmarks.landmark into a series of x,y,z

    Arguments:
        landmark -- hand_landmarks.landmark

    Keyword Arguments:
        order -- whether to return it sorted by point so (x1, y1, z1) then (x2, y2, z2) (default: {False})

    Returns:
        a Series of 21 * 3 values for the hand_landmark
    """
    x_ls = []
    y_ls = []
    z_ls = []
    
    if order:
        ordered = []
    for p in landmark:
        p = str(p).split('\n')[:-1]
        x = float(p[0].split(' ')[-1])
        y = float(p[1].split(' ')[-1])
        z = float(p[2].split(' ')[-1])
        
        if order:
            ordered.extend([x,y,z])
        else:
            x_ls.append(x)    
            y_ls.append(y)    
            z_ls.append(z)    
    if order:
        result = ordered
    else:
        result = [*x_ls, *y_ls, *z_ls]
    
    return pd.Series(result)

def normalize_hand(x: pd.Series, with_label=True) -> pd.Series:
    """takes a row and returns a normalized row with $(x_i, y_i) = [(x_i - x_0, y_i - y_0)]$ 
    then dividing them by y12 $(x_i, y_i) = (x_i/y_12, y_i/y_12)

    Arguments:
        x           -- row as a pd.series with data of 21 point + label
        with_label  -- Bolean value indicating whether to return the label or not

    Returns:
        hand marks normalized 
    """
    xs = (x[0:-1:3] - x.iloc[0])
    ys = (x[1:-1:3] - x.iloc[1])
    xs = xs/ys.iloc[12]
    ys = ys/ys.iloc[12]
    zs =  x[2:-1:3]
    
    if with_label:
        label = x[-1:-2:-1]
        result = pd.concat([xs, ys, zs, label])
    
    else:
        result = pd.concat([xs, ys, zs])
   
    return result
    
    
    
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
last5 = {}


cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                hand_x = int(min(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
                hand_y = int(min(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0]) - 20
               
                tst = get_tst_points(hand_landmarks.landmark, order=True)
                tst = normalize_hand(tst)
                
                pred = model.predict(tst.values.reshape((-1, 63)))
                pred = encoder.inverse_transform(pred)
                
                if i not in last5:
                    last5[i] = []
                    
                last5[i].append(pred[0])
                if len(last5[i]) > 5:
                    last5[i].pop(0)
                    
                label = st.mode(last5[i])
                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                
                text_x = max(0, min(hand_x, frame.shape[1] - text_size[0]))
                text_y = max(30, hand_y)
                
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                              (text_x + text_size[0] + 5, text_y + 5), 
                              (0, 0, 0), -1)
                
                cv2.putText(frame, label, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # cv2.imwrite('tst.png', frame)
                
        cv2.imshow("Hand Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()