from tkinter import Tk, StringVar, OptionMenu, Button, Entry, Label, messagebox, ttk
from tkinter import *
from threading import Thread
import cv2
import time
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
from pygame import mixer
import os
import sys
import pygetwindow as gw


fps = 0
frame_counter = 0
start_time_fps = time.time()
show_fps = True

mixer.init()
background_music_file = "C:\\Users\\ismet\\Desktop\\proje\\bg_music.mp3"
mixer.music.load(background_music_file)
mixer.music.play(-1)


audio_files = {
    0:"C:\\Users\\ismet\\Desktop\\proje\\en_num_0.mp3",
    1:"C:\\Users\\ismet\\Desktop\\proje\\en_num_01.mp3",
    2:"C:\\Users\\ismet\\Desktop\\proje\\en_num_02.mp3",
    3:"C:\\Users\\ismet\\Desktop\\proje\\en_num_03.mp3",
    4:"C:\\Users\\ismet\\Desktop\\proje\en_num_04.mp3",
    5:"C:\\Users\\ismet\\Desktop\\proje\\en_num_05.mp3",
    6:"C:\\Users\\ismet\\Desktop\\proje\\en_num_06.mp3",
    7:"C:\\Users\\ismet\\Desktop\\proje\\en_num_07.mp3",
    8:"C:\\Users\\ismet\\Desktop\\proje\\en_num_08.mp3",
    9:"C:\\Users\\ismet\\Desktop\\proje\\en_num_09.mp3",
    10:"C:\\Users\\ismet\\Desktop\\proje\\en_num_10.mp3",
}


# Mediapipe Hands module setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def start_video():
    global finger_counts, start_time, last_saved_time, camera_video
    finger_counts = []
    start_time = None
    last_saved_time = None
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,1280)
    camera_video.set(4,960)

class HomePage:

    def __init__(self, root):
        self.root = root
        self.root.geometry('1920x1080')
        self.root.attributes('-fullscreen', True)
        image_path = 'C:\\Users\\ismet\\Desktop\\proje\\8122814.png'
        image = Image.open(image_path)
        image = image.resize((1920, 1080), Image.ANTIALIAS)
        self.background_image = ImageTk.PhotoImage(image)
        background_label = Label(self.root, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        Button(self.root, text="Start", command=self.start, font=("Comic Sans MS", 30), bg='green', activebackground='light green', fg='white', bd=10, relief='raised').place(relx=0.5, rely=0.5, anchor=CENTER)
        Button(self.root, text="Quit", command=self.quit, font=("Comic Sans MS", 30), bg='green', activebackground='light green', fg='white', bd=10, relief='raised').place(relx=0.5, rely=0.7, anchor=CENTER)


    def start(self):
        self.root.withdraw()
        operation_window = Toplevel(self.root)
        OperationPage(operation_window)

    def quit(self):
        self.root.quit()


class OperationPage:

    def __init__(self, root):
        self.root = root
        self.root.geometry('1920x1080')
        self.root.attributes('-fullscreen', True)
        image_path = 'C:\\Users\\ismet\\Desktop\\proje\\operations_bg.png'
        self.root.title("Finger Counting")
        image = Image.open(image_path)
        image = image.resize((1920, 1080), Image.ANTIALIAS)
        self.background_image = ImageTk.PhotoImage(image)
        background_label = Label(self.root, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        Button(self.root, text="Addition", command=lambda: self.select_operation("Addition"), font=("Comic Sans MS", 20), bg='yellow', activebackground='light yellow', fg='black', bd=5, relief='raised').place(relx=0.5, rely=0.4, anchor=CENTER)
        Button(self.root, text="Subtraction", command=lambda: self.select_operation("Subtraction"), font=("Comic Sans MS", 20), bg='yellow', activebackground='light yellow', fg='black', bd=5, relief='raised').place(relx=0.5, rely=0.5, anchor=CENTER)
        Button(self.root, text="Multiplication", command=lambda: self.select_operation("Multiplication"), font=("Comic Sans MS", 20), bg='yellow', activebackground='light yellow', fg='black', bd=5, relief='raised').place(relx=0.5, rely=0.6, anchor=CENTER)
        Button(self.root, text="Division", command=lambda: self.select_operation("Division"), font=("Comic Sans MS", 20), bg='yellow', activebackground='light yellow', fg='black', bd=5, relief='raised').place(relx=0.5, rely=0.7, anchor=CENTER)
        Button(self.root, text="Back", command=lambda: self.select_operation("Division"), font=("Comic Sans MS", 20), bg='green', activebackground='light green', fg='black', bd=5, relief='raised').place(relx=0.5, rely=0.77, anchor=CENTER)




    def select_operation(self, operation):
        global math_operation
        math_operation = operation
        self.root.destroy()
        video_thread = Thread(target=start_video)
        video_thread.start()


def reset_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)



class UserInputPage:

    def reset(self):
        self.root.destroy()
        reset_program()

    def check_answer(self):
        user_answer = self.entry.get()
        self.root.destroy()
        if len(finger_counts) > 1:
            if math_operation == "Addition":
                correct_answer = sum(finger_counts)
            elif math_operation == "Subtraction":
                correct_answer = finger_counts[0] - sum(finger_counts[1:])
            elif math_operation == "Multiplication":
                correct_answer = np.prod(finger_counts)
            elif math_operation == "Division":
                correct_answer = finger_counts[0] / np.prod(finger_counts[1:])
            if float(user_answer) == correct_answer:
                correct_sound = mixer.Sound(correct_answer_file)
                correct_sound.play()
                messagebox.showinfo("Result", "It's correct!")
            else:
                wrong_sound = mixer.Sound(wrong_answer_file)
                wrong_sound.play()
                messagebox.showerror("Result", "It's incorrect. Try again.")


correct_answer_file = "C:\\Users\\ismet\\Desktop\\proje\\correct.mp3"
wrong_answer_file = "C:\\Users\\ismet\\Desktop\\proje\\wrong.mp3"


# Mediapipe functions
def detectHandsLandmarks(image, hands, draw=True):
    output_image = image.copy()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks and draw:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image = output_image, landmark_list = hand_landmarks,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),
                                                                                     thickness=2, circle_radius=2))
    return output_image, results

def countFingers(image, results, draw=True):
    height, width, _ = image.shape
    output_image = image.copy()
    count = {'RIGHT': 0, 'LEFT': 0}
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_label = hand_info.classification[0].label
        hand_landmarks =  results.multi_hand_landmarks[hand_index]
        for tip_index in fingers_tips_ids:
            finger_name = tip_index.name.split("_")[0]
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                count[hand_label.upper()] += 1
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
        if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            count[hand_label.upper()] += 1
    return output_image, fingers_statuses, count

def start_gui():
    root = Tk()
    HomePage(root)
    root.mainloop()

def check_answer_gui(correct_answer):
    def check_answer():
        user_answer = entry.get()
        if float(user_answer) == correct_answer:
            correct_sound = mixer.Sound(correct_answer_file)
            correct_sound.play()
            messagebox.showinfo("Result", "It's correct!")
        else:
            wrong_sound = mixer.Sound(wrong_answer_file)
            wrong_sound.play()
            messagebox.showerror("Result", "It's incorrect. Try again.")

    root = Tk()
    root.geometry('1920x1080')
    root.attributes('-fullscreen', True)
    root.configure(bg='green')
    Label(root, text="Enter your answer:", font=("Comic Sans MS", 32), bg='green').pack(pady=20)
    entry = Entry(root, font=("Helvetica", 24))
    entry.pack()
    Button(root, text="Submit", command=check_answer, font=("Comic Sans MS", 20), bg='yellow').pack(pady=10)
    Button(root, text="Back", command=check_answer, font=("Comic Sans MS", 20), bg='yellow').pack(pady=15)
    Button(root, text="Quit", command=root.destroy, font=("Comic Sans MS", 20), bg='yellow', activebackground='light yellow', fg='black', bd=5, relief='raised').place(relx=0.5, rely=0.7, anchor=CENTER)

    root.mainloop()

if __name__ == "__main__":
    gui_thread = Thread(target=start_gui)
    gui_thread.start()

# Finger counting
finger_counts = []
start_time = None
last_saved_time = None
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

while camera_video.isOpened():
    frame_start_time = time.time()
    
    ok, frame = camera_video.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    frame, results = detectHandsLandmarks(frame, hands_videos, draw=False)
    if results.multi_hand_landmarks:
        if start_time is None:
            start_time = time.time()
        if time.time() - start_time > 3:  # wait 2 seconds before processing
            _, _, count = countFingers(frame, results, draw=False)
            current_sum = sum(count.values())
            if current_sum not in finger_counts:
                finger_counts.append(current_sum)
                # Play the corresponding audio file
            if current_sum in audio_files and os.path.isfile(audio_files[current_sum]):
                finger_sound = mixer.Sound(audio_files[current_sum])
                finger_sound.play()

                start_time = None  # reset start_time here
                last_saved_time = time.time()
        elif last_saved_time is None or time.time() - last_saved_time < 3:
            frame, _, _ = countFingers(frame, results, draw=True)
    else:
        if start_time is not None and time.time() - start_time > 6:
            start_time = None  # reset start_time here
            break

    frame_counter += 1
    elapsed_time = time.time() - start_time_fps
    if elapsed_time > 1:  # every second
        fps = frame_counter / elapsed_time
        frame_counter = 0
        start_time_fps = time.time()

    if show_fps:
        frame = cv2.putText(frame, f"FPS: {fps:.2f}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 105, 180), 2)

    if last_saved_time is not None and time.time() - last_saved_time < 3:
        frame = cv2.putText(frame, "Number is saved, please enter another number", (10, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 105, 180), 2)
    if finger_counts:
        frame = cv2.putText(frame, ' '.join(map(str, finger_counts)), (10, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 105, 180), 2)
    cv2.imshow('Fingers Counter', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Perform the math operation and check the answer
if len(finger_counts) > 1:
    if math_operation == "Addition":
        correct_answer = sum(finger_counts)
    elif math_operation == "Subtraction":
        correct_answer = finger_counts[0] - sum(finger_counts[1:])
    elif math_operation == "Multiplication":
        correct_answer = np.prod(finger_counts)
    elif math_operation == "Division":
        correct_answer = finger_counts[0] / np.prod(finger_counts[1:])
    check_answer_gui(correct_answer)

camera_video.release()
cv2.destroyAllWindows()    
mixer.quit()  # Shut down the mixer