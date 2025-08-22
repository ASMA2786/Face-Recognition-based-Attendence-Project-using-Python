import os
import pickle
import cv2
import numpy as np

faces_file = 'data/faces_data.pkl'
names_file = 'data/names.pkl'

if not os.path.exists(faces_file) or not os.path.exists(names_file):
    print("No faces found")
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)

    with open(names_file, 'rb') as f:
        names = pickle.load(f)

    if len(names) == 0:
        print("No faces found")
    else:
        print("Faces shape:", faces.shape)  # e.g., (100, 7500)
        print("Names length:", len(names))
        print("First 5 names:", names[:5])

        # Optional: display first 5 faces
        for i in range(min(5, len(names))):
            face_img = faces[i].reshape(50, 50, 3)
            cv2.imshow(f"Face {i+1} - {names[i]}", face_img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
