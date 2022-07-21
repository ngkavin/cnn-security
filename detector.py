"""
Program for operating a rudimentary security system
on the Raspberry Pi 4. Requires a PIR sensor and camera (legacy stack).
Also requires a TorchScript model at MODEL_PATH and a PushBullet access token
under TOKEN_PATH.

Runs until interrupted. If motion is detected, the camera
passes what it sees into the nerual network. If an authorized user
is not detected within a specified time (default 20 seconds), then
an alert is sent to the owner via PushBullet.

Writen by Kavin Nguyen for EE P 522.
"""

import os
import subprocess
import socket
import time

import cv2
import torch
from torchvision import transforms
from PIL import Image

from pushbullet import PushBullet
import pigpio

PIR = 14                                    # GPIO of PIR sensor
MAX_DETECTION_DURATION = 2000               # Max time to perform detection (sec)
DETECTION_DELAY = 100                       # Min amount of time between alerts (sec)
TOKEN_PATH = 'secrets/pushbullet.txt'       # PushBullet access token path
MODEL_PATH = 'model/en-b0-epoch01_ts.pt'    # Path to TorchScript model
CLASSES = ['Unauthorized', 'User1']         # Detection types in the model

# Convert image to tensor and resize to 224 (mini-batch of 1)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224))
])


def get_videocapture():
    """Initialize the camera

    Returns
    -------
    cv2 VideoCapture object
        image stream from the Raspberry Pi camera
    """
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv2.CAP_PROP_FPS, 36)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 80)
    cap.set(cv2.CAP_PROP_CONTRAST, 100)
    cap.set(cv2.CAP_PROP_SATURATION, 30)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 5000)
    return cap


def read_from_camera(cap):
    """Get an image from the camera

    Parameters
    ----------
    cap : cv2 VideoCapture object
        Object that camera images are pulled from

    Returns
    -------
    numpy array
        RGB image
    """
    ret, image = cap.read()
    if not ret:
        raise RuntimeError("Error reading frame")
    # convert BGR image to RGB
    return image[:, :, [2, 1, 0]]


def save_image(cap):
    """Save an image to disk as "alert_image.jpg"

    Parameters
    ----------
    cap : cv2 VideoCapture object
        Object that camera images are pulled from
    """
    saved_img = read_from_camera(cap)
    # Save low-quality image
    im = Image.fromarray(saved_img).resize((128, 128))
    im.save("alert_image.jpg", quality=20, optimize=True)


def run_detection(cap):
    """Run facial detection

    Parameters
    ----------
    cap : cv2 VideoCapture object
        Object that camera images are pulled from

    Returns
    -------
    bool
        An authorized user was detected
    """
    model = torch.jit.load(MODEL_PATH).eval()

    duration = time.time() + MAX_DETECTION_DURATION
    with torch.no_grad():
        while time.time() < duration:

            img = read_from_camera(cap)
            img = preprocess(img)[None]

            output = model(img)
            top = list(enumerate(output[0].softmax(dim=0)))
            top.sort(key=lambda x: x[1], reverse=True)
            for idx, val in top:
                print(f"{val.item()*100:.2f}% {CLASSES[idx]}")

            # Stop collecting more images if higher chance of user than unauthorized
            if top[0][0] > 0:
                return True

    return False


def send_pushbullet_alert(access_token, detection_time):
    """ Send an alert through PushBullet

    Parameters
    ----------
    access_token : str
        PushBullet access token
    detection_time : float
        timestamp frome time.time()
    """
    pb = PushBullet(access_token)
    pb.push_note("Motion Detected", detection_time.strftime("%d/%m/%Y %H:%M:%S"))
    with open('alert_image.jpg', 'rb') as pic:
        file_data = pb.upload_file(pic, 'alert_image.jpg')
    pb.push_file(**file_data)
    os.remove('alert_image.jpg')


def alert(gpio, level, tick):
    """Alert callback function.
    Preforms detection and sends alert if an authorzied user is not detected.

    Parameters
    ----------
    gpio : int
        GPIO that triggered callback. Required by pigpio
    level : int
        indicates rising/falling edge. Required by pigpio
    tick : int
        tick number that the pigpio daemon keeps track of
    """
    print(f"Detection on GPIO {gpio}")
    # Get timestamp
    detection_time = time.time()
    # Initialize the camera
    cap = get_videocapture()
    # Save the first thing the camera sees
    save_image(cap)
    end_time = time.time()
    print(f'capture time: {int(round((detection_time-end_time) * 1000))}')
    # Check if an authorized user is detected
    authorized = run_detection(cap)
    if not authorized:
        with open(TOKEN_PATH, 'r') as f:
            access_token = f.readlines()[0]
        print("Sending intruder alert...")
        send_pushbullet_alert(access_token, detection_time)
    # Prevent redetection for a specified time
    time.sleep(DETECTION_DELAY)


if __name__ == '__main__':
    # Start the pigpio daemon if it is not already started
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        disbale_daemon = False
        if s.connect_ex(('localhost', 8888)) != 0:
            process = subprocess.Popen("sudo pigpiod".split(), stdout=subprocess.PIPE)
            process.communicate()
            time.sleep(1)
            disbale_daemon = True

    pi = pigpio.pi()
    pi.set_mode(PIR, pigpio.INPUT)
    pi.callback(PIR, pigpio.RISING_EDGE, alert)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print('Interrupted - Shutting down pigpio')
    pi.stop()

    # Kill the daemon if it was started by this script
    if disbale_daemon:
        subprocess.Popen("sudo killall pigpiod".split(), stdout=subprocess.PIPE)
        process.communicate()
