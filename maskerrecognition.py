import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 105

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 40)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Terima kasih sudah menggunakan masker"
not_weared_mask = "INGAT PESAN DOI PAKAI MASKERMU"

# Read video
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

while 1:
    # Get individual frame
    ret, img = cap.read()
    img = cv2.flip(img,1)

    # Convert Image into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image in black and white
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('black_and_white', black_and_white)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    # Face prediction for black and white
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.5, 5)


    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face found...", org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
        # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        # Draw rectangle on gace
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect lips counters
            eyes = eye_cascade .detectMultiScale(gray, 1.5, 5)

        # Face detected but Lips not detected which means person is wearing mask
        if(len(eyes) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
            # cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in eyes:

                if(y < my < y + h):
                    # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                    # person is not waring mask
                    cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)
                    # cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
                    # cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                    break

    # Show frame with results
    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
cv2.destroyAllWindows()
