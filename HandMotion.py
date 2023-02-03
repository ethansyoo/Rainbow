import cv2
from cv2 import waitKey
from matplotlib.pyplot import draw
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#once sign language is fine-tuned, can possibly create a way to spell and print words on the screen using sign language

letters = []

def spell(char):
  if cv2.waitKey(13) == ord('Enter'):
    letters.append(char)
    print("Appended")
  if cv2.waitKey(32) == ord("Space"):
    letters.append(' ')
    print("Space")
spell('A')
def delete():
  letters.pop()


def directional_angle(x1, y1, x2, y2):
  return math.degrees(math.atan2(y1 - y2, x1 - x2))

def angle(x1, y1, x2, y2, refx, refy): #returns the angle of the reference point given two other points using vectors to calculate the angle between the vectors
  lst1 = [x1 - refx, y1 - refy]
  lst2 = [x2 - refx, y2 - refy]
  norm1 = lst1 / np.linalg.norm(lst1)
  norm2 = lst2 / np.linalg.norm(lst2)
  dot = np.dot(norm1, norm2)
  return math.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

def angles_between(x1, y1, refx, refy, x2, y2, min, max): #returns boolean if finger angles are between the given thresholds
  return angle(x1, y1, x2, y2, refx, refy) < max and angle(x1, y1, x2, y2, refx, refy) > min

def finger_slope(y1, y2, x1, x2): #finds the slope of the points
  return (y1 - y2) / (x1 - x2)

def straight_line(tip_x, tip_y, dip_x, dip_y, pip_x, pip_y, mcp_x, mcp_y, thresh): #if the given points in the finger are in a straight line within the given threshold, then returns true, if not false
  return (abs((tip_y - pip_y) / (tip_x - pip_x)) < abs((tip_y - mcp_y) / (tip_x - mcp_x)) + thresh and abs((tip_y - pip_y) / (tip_x - pip_x)) > abs((tip_y - mcp_y) / (tip_x - mcp_x)) - thresh or (tip_y < mcp_y + thresh and tip_y > mcp_y - thresh and tip_y < pip_y + thresh and tip_y > pip_y - thresh) or (tip_x < pip_x + thresh and tip_x > pip_x - thresh and tip_x < mcp_x + thresh and tip_x > mcp_x - thresh) and not (tip_x < wrist_x and tip_x > dip_x) and not (tip_x > wrist_x and tip_x < dip_x))

def straight_angle(tip_x, tip_y, pip_x, pip_y, mcp_x, mcp_y, thresh): #if the given points are within the threshold of +- 180 degrees return true, else false
  return angle(tip_x, tip_y, mcp_x, mcp_y, pip_x, pip_y) < 180 + thresh and angle(tip_x, tip_y, mcp_x, mcp_y, pip_x, pip_y) > 180 - thresh
# not (tip_x < wrist_x and tip_x > dip_x) and not (tip_x > wrist and tip_x < dip_x)

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    


    image_height, image_width, _ = image.shape
    if (results.multi_hand_landmarks):
      for hand_landmarks in results.multi_hand_landmarks:
        i_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        i_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        i_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
        i_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
        i_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
        i_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        i_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        i_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        r_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
        r_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        r_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x
        r_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
        r_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
        r_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
        r_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
        r_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
        p_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
        p_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        p_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x
        p_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y
        p_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x
        p_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
        p_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
        p_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
        t_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        t_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        t_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
        t_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
        t_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
        t_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
        t_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x
        t_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y
        straight_index = straight_line(i_tip_x, i_tip_y, i_dip_x, i_dip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 1.25) and i_tip_y < i_pip_y
        m_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
        m_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        m_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
        m_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
        m_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
        m_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        m_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
        m_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        straight_middle = straight_line(m_tip_x, m_tip_y, m_dip_x, m_dip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 0.075) and m_tip_y < m_pip_y
        straight_ring = straight_line(r_tip_x, r_tip_y, r_dip_x, r_dip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 0.075) and r_tip_y < r_pip_y
        straight_pinky = straight_line(p_tip_x, p_tip_y, p_dip_x, p_dip_y, p_pip_x, p_pip_y, p_mcp_x, p_mcp_y, 0.075) and p_tip_y < p_pip_y
        straight_thumb = straight_line(t_tip_x, t_tip_y, t_dip_x, t_dip_y, t_pip_x, t_pip_y, t_mcp_x, t_mcp_y, 0.05) and t_tip_y < t_pip_y and not (t_tip_x > p_mcp_x and t_tip_x < t_mcp_x) and not (t_tip_x < p_mcp_x and t_tip_x > t_mcp_x)
        print('hand_landmarks:', hand_landmarks)
        print(
            f'Thumb finger tip coordinates: (',
            f'{t_tip_x}, '
            f'{t_tip_y})'
            )
        # print(
        #     f'Index finger dip coordinates: (',
        #     f'{i_dip_x}, '
        #     f'{i_dip_y})'
        #     )
        # print(
        #     f'Index finger pip coordinates: (',
        #     f'{i_pip_x}, '
        #     f'{i_pip_y})'
        #     )
        print(
            f'Thumb cmc: (',
            f'{t_mcp_x}, '
            f'{t_mcp_y})'
            )
        print(
          f'Pinky cmc: (',
          f'{p_mcp_x}, '
          f'{p_mcp_y})'
          )
        print(
          f'Angle: (',
          f'{math.degrees(math.atan2((i_tip_y - i_dip_y), (i_tip_x - i_dip_x)))})'
          ) #gives the angle of the line between two points relative to the screen.
        print(
          f'Two point angle: (',
          f'{angle(t_tip_x, t_tip_y, t_mcp_x, t_mcp_y, t_pip_x, t_pip_y)}, '
          f'{angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y, i_pip_x, i_pip_y)}, '
          f'{angle(m_tip_x, m_tip_y, m_mcp_x, m_mcp_y, m_pip_x, m_pip_y)}, '
          f'{angle(r_tip_x, r_tip_y, r_mcp_x, r_mcp_y, r_pip_x, r_pip_y)}, '
          f'{angle(p_tip_x, p_tip_y, p_mcp_x, p_mcp_y, p_pip_x, p_pip_y)}, '
          f'{angle(i_tip_x, i_tip_y, wrist_x, wrist_y, i_mcp_x, i_mcp_y)}, '
          f'{angle(i_tip_x, i_tip_y, wrist_x, wrist_y, t_tip_x, t_tip_y)})'
          f'\n',
          f'Direction: ',
          f'{directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y)}'
          f'\n',
          f'I and M Angle: ',
          f'{angle(i_tip_x, i_tip_y, m_tip_x, m_tip_y, wrist_x, wrist_y)}'

        )
        # if (straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 10)):
        #   cv2.putText(image, "Angle", (640, 480), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)

        #should be noted that since the image will be reversed, the y is also inversed, so y1 < y2 is actually y1 > y2
        if (straight_index): #debugging lines to help signal which fingers and functions are being picked up
          cv2.putText(image, "Index", (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4) 
        if (straight_middle):
          cv2.putText(image, "Middle", (50, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
        if (straight_ring):
          cv2.putText(image, "Ring", (50, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
        if (straight_pinky):
          cv2.putText(image, "Pinky", (50, 170), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
        if (straight_thumb):
          cv2.putText(image, "Thumb", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
        if (straight_line(t_tip_x, t_tip_y, t_dip_x, t_dip_y, t_pip_x, t_pip_y, t_mcp_x, t_mcp_y, 1.1) and t_tip_y < t_dip_y): #debugging line
          cv2.putText(image, "Straight Thumb", (50, 230), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)

        if (straight_index and not straight_middle and not straight_ring and not straight_pinky and not straight_thumb): #all the functions from the if statements are to implement algorithms to identify the various types of hand positions
          cv2.putText(image, "One", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (straight_index and straight_middle and not straight_ring and not straight_pinky and not straight_thumb):
          cv2.putText(image, "Two", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (straight_index and straight_middle and straight_ring and not straight_pinky and not straight_thumb):
          cv2.putText(image, "Three", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (straight_index and straight_middle and straight_ring and straight_pinky and not straight_thumb):
          cv2.putText(image, "Four", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (straight_index and straight_middle and straight_ring and straight_pinky and straight_thumb):
          cv2.putText(image, "Five", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (straight_pinky and straight_thumb and not straight_middle and not straight_ring and not straight_index):
          cv2.putText(image, "Shaka", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (straight_line(t_tip_x, t_tip_y, t_dip_x, t_dip_y, t_pip_x, t_pip_y, t_mcp_x, t_mcp_y, 1.2) and t_tip_y < t_dip_y and not straight_index and not straight_middle and not straight_ring and not straight_pinky):
          cv2.putText(image, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (straight_line(t_tip_x, t_tip_y, t_dip_x, t_dip_y, t_pip_x, t_pip_y, t_mcp_x, t_mcp_y, 1.2) and not straight_index and not straight_middle and not straight_ring and not straight_pinky and t_tip_y > t_dip_y):
          cv2.putText(image, "Thumbs Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (not straight_thumb and not straight_index and straight_ring and straight_middle and straight_pinky):
          cv2.putText(image, "Okay", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (straight_thumb and straight_index and straight_pinky and not straight_ring and not straight_middle):
          cv2.putText(image, "Rock & Roll", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        elif (straight_thumb and straight_index and not straight_ring and not straight_middle and not straight_pinky):
          cv2.putText(image, "Loser", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        else:
          cv2.putText(image, "None", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)

        #sign language alphabet identifier
        if (angles_between(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 0, 40) and angles_between(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 0, 40) and angles_between(r_tip_x, r_tip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 0, 40) and t_tip_x < i_pip_x and t_tip_y < i_pip_y):
          cv2.putText(image, "A", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 10) and straight_angle(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 10) and straight_angle(r_tip_x, r_tip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 10) and straight_angle(p_tip_x, p_tip_y, p_pip_x, p_pip_y, p_mcp_x, p_mcp_y, 10) and not straight_thumb):
          cv2.putText(image, "B", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y, i_pip_x, i_pip_y) < 160 and angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y, i_pip_x, i_pip_y) > 90 and angle(m_tip_x, m_tip_y, m_mcp_x, m_mcp_y, m_pip_x, m_pip_y) < 160 and angle(m_tip_x, m_tip_y, m_mcp_x, m_mcp_y, m_pip_x, m_pip_y) > 90 and angle(r_tip_x, r_tip_y, r_mcp_x, r_mcp_y, r_pip_x, r_pip_y) < 160 and angle(r_tip_x, r_tip_y, r_mcp_x, r_mcp_y, r_pip_x, r_pip_y) > 90 and angle(p_tip_x, p_tip_y, p_mcp_x, p_mcp_y, p_pip_x, p_pip_y) < 160):
          cv2.putText(image, "C", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (not straight_middle and not straight_ring and not straight_pinky and not straight_thumb and 
        straight_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y, wrist_x, wrist_y, 15)
        and not straight_angle(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 10)
        and angles_between(i_tip_x, i_tip_y, t_tip_x, t_tip_y, wrist_x, wrist_y, 145, 185)):
          cv2.putText(image, "D", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (t_tip_y > p_dip_y and t_tip_y > r_dip_y 
        and t_tip_y > m_dip_y and t_tip_y > i_dip_y 
        and not straight_thumb and not straight_index 
        and not straight_ring and not straight_middle 
        and not straight_pinky and angles_between(p_tip_x, p_tip_y, p_pip_x, p_pip_y, p_mcp_x, p_mcp_y, 0, 40) 
        and angles_between(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 0, 30) 
        and angles_between(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 0, 30) 
        and angles_between(r_tip_x, r_tip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 0, 30)):
          cv2.putText(image, "E", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (straight_pinky and straight_ring and straight_middle and not straight_index and not straight_thumb):
          cv2.putText(image, "F", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 15) 
        and angles_between(t_tip_x, t_tip_y, t_pip_x, t_pip_y, t_mcp_x, t_mcp_y, 145, 180)
        and angles_between(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y, wrist_x, wrist_y, 100, 130)
        and (directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y) > 170 or directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y) < -170)):
          cv2.putText(image, "G", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 15) 
        and straight_angle(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 10) 
        and straight_index and not straight_ring and not straight_pinky 
        and angles_between(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y, wrist_x, wrist_y, 100, 150)):
          cv2.putText(image, "H", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (straight_angle(p_tip_x, p_tip_y, p_pip_x, p_pip_y, p_mcp_x, p_mcp_y, 10) and not straight_angle(t_tip_x, t_tip_y, t_pip_x, t_pip_y, t_mcp_x, t_mcp_y, 10) and not straight_ring and not straight_middle and not straight_index and not straight_thumb):
          cv2.putText(image, "I", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #need one for J figure out how to address movement
        elif (straight_index and straight_middle and not straight_ring and not straight_pinky and t_tip_x > i_tip_x 
        and t_tip_x < m_tip_x and directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y) < -70 
        and directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y) > -110 and t_tip_y < r_pip_y):
          cv2.putText(image, "K", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (straight_index and not straight_middle and not straight_ring and not straight_pinky 
        and angles_between(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y, wrist_x, wrist_y, 150, 190) 
        and straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 10)
        and angles_between(i_tip_x, i_tip_y, t_tip_x, t_tip_y, wrist_x, wrist_y, 100, 130)
        and directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y) > -110 and directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y) < -70):
          cv2.putText(image, "L", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (angles_between(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 0, 40) and angles_between(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 0, 40) and angles_between(r_tip_x, r_tip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 0, 40) and t_tip_x > r_pip_x and t_tip_x < p_pip_x):
          cv2.putText(image, "M", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (angles_between(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 0, 40) and angles_between(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 0, 40) and angles_between(r_tip_x, r_tip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 0, 40) and t_tip_x > m_pip_x and t_tip_x < r_pip_x and t_tip_y < m_pip_y):
          cv2.putText(image, "N", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #need one for O, O is just a closer C
        elif (angles_between(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 40, 80) and angles_between(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 40, 70) and angles_between(r_tip_x, r_tip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 40, 70)):
          cv2.putText(image, "O", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #fix and base on angles for P
        elif (straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 10) and straight_angle(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 20) 
        and i_tip_x < t_tip_x and m_tip_x < t_tip_x and directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y) > 0):
          cv2.putText(image, "P", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #Q is the same as G, except downwards
        elif (straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 10) and angles_between(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y, wrist_x, wrist_y, 90, 130) and angles_between(i_tip_x, i_tip_y, t_tip_x, t_tip_y, wrist_x, wrist_y, 100, 140) and not straight_middle and not straight_ring and not straight_pinky and directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y) > 70 and directional_angle(i_tip_x, i_tip_y, i_mcp_x, i_mcp_y) < 110):
          cv2.putText(image, "Q", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #account for crossing/intersecting lines for R
        elif (straight_index and straight_middle and not straight_ring and not straight_pinky and not straight_thumb and i_tip_x > m_tip_x):
          cv2.putText(image, "R", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #S is similar to E and A, must find distinction between the three
        elif (angles_between(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 0, 40) and angles_between(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 0, 40) and angles_between(r_tip_x, r_tip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 0, 40) and t_tip_y > m_pip_y and not straight_angle(t_tip_x, t_tip_y, t_pip_x, t_pip_y, t_mcp_x, t_mcp_y, 10)):
          cv2.putText(image, "S", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #account for the thumb inbetween the fingers
        elif (angles_between(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 0, 40) and angles_between(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 0, 40) and angles_between(r_tip_x, r_tip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 0, 40) and t_tip_x > i_pip_x and t_tip_x < m_pip_x and t_tip_y < m_pip_y):
          cv2.putText(image, "T", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #U is an upright H
        elif (straight_index and straight_middle and not straight_ring and not straight_pinky and not straight_thumb and angles_between(i_tip_x, i_tip_y, wrist_x, wrist_y, m_tip_x, m_tip_y, 0, 5)):
          cv2.putText(image, "U", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #find distinction between U and V, U is next to each other, V is apart
        elif (straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 10) and straight_angle(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 10) and not straight_ring and not straight_pinky and not straight_thumb and angles_between(i_tip_x, i_tip_y, wrist_x, wrist_y, m_tip_x, m_tip_y, 5, 20)):
          cv2.putText(image, "V", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 10) and straight_angle(m_tip_x, m_tip_y, m_pip_x, m_pip_y, m_mcp_x, m_mcp_y, 10) and straight_angle(r_tip_x, r_tip_y, r_pip_x, r_pip_y, r_mcp_x, r_mcp_y, 10) and not straight_angle(p_tip_x, p_tip_y, p_pip_x, p_pip_y, p_mcp_x, p_mcp_y, 10) and not straight_thumb):
          cv2.putText(image, "W", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #need one for X
        elif (angles_between(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 75, 155) and not straight_angle(i_tip_x, i_tip_y, i_pip_x, i_pip_y, i_mcp_x, i_mcp_y, 25) and not straight_thumb and not straight_ring and not straight_middle and not straight_pinky):
          cv2.putText(image, "X", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        elif (straight_angle(p_tip_x, p_tip_y, p_pip_x, p_pip_y, p_mcp_x, p_mcp_y, 10) and straight_angle(t_tip_x, t_tip_y, t_pip_x, t_pip_y, t_mcp_x, t_mcp_y, 10) and not straight_middle and not straight_index and not straight_ring):
          cv2.putText(image, "Y", (640, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        #need one for Z

        
    
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()