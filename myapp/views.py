import os
import cv2
import base64
import json
import pandas as pd
from django.shortcuts import render
import numpy as np
import pickle
from django.http import JsonResponse
import torch
from django.http import HttpResponse
from sklearn.neighbors import NearestNeighbors
from facenet_pytorch import InceptionResnetV1

Face_data = pd.DataFrame(columns=['Name', 'Embeddings'])
Person_data = pd.DataFrame(columns=['Name', 'Image'])
Images_preprocessed = []
person_face = 0
flann = NearestNeighbors(n_neighbors=1, algorithm='auto')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = BASE_DIR + "/staticfiles/haarcascade_face.xml"
trainy = Face_data['Name']
resnet = InceptionResnetV1().eval()
state_dict = torch.load(BASE_DIR + "/staticfiles/model.pth")
resnet.load_state_dict(state_dict, strict=False)

face_cascade = cv2.CascadeClassifier(path)


def home(request):
    return render(request, 'home.html')


def index(request):
    return render(request, 'start_capture.html')


def face_add(request):
    global Images_preprocessed
    Images_preprocessed = []
    return render(request, 'Add_face.html')


def save_dataframe(request):
    global Face_data, Person_data, flann
    # Assuming you have a pandas DataFrame called 'Face_data'
    # You can customize the saving logic based on your requirements

    combined_dataframes = [Face_data, Person_data, flann]
    pickle_data = pickle.dumps(combined_dataframes)

    # Create an HttpResponse with the CSV data
    response = HttpResponse(content_type='application/octet-stream')
    response['Content-Disposition'] = 'attachment; filename="Face_data.pkl"'
    response.write(pickle_data)

    return response


def face_recognize_page(request):
    if Face_data.empty:
        message = "No data available"
        return render(request, 'start_capture.html', {'message': message})
    else:
        return render(request, 'recognize.html')


def data_base(request):
    return render(request, 'Data Load.html')


def dataloader(request):
    global Face_data, Person_data, flann, trainy

    if request.method == 'POST':
        print('DONE')

        file = request.FILES['data_file']

        if not file:
            return JsonResponse({'status': 'error', 'message': 'No file uploaded'})

        if not file.name.endswith('.pkl'):
            return JsonResponse({'status': 'error', 'message': 'Invalid file format'})

        try:

            loaded_dataframes = pd.read_pickle(file)
            Face_data = loaded_dataframes[0]
            Person_data = loaded_dataframes[1]
            flann = loaded_dataframes[2]
            trainy = Face_data['Name']
            print("loading")

            return JsonResponse({'status': 'success', 'message': 'PKL file loaded successfully'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': 'Error loading PKL file: ' + str(e)})

    return render(request, 'Data Load.html')


def recognize_fun(request):
    global face_cascade

    if request.method == 'POST':
        payload = json.loads(request.body)
        image_data = payload.get('image', [])
        print('Image Recieved')

        # Convert the base64 encoded image data to OpenCV format
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:

            # Process the detected faces as needed
            for (x, y, w, h) in faces:
                # Perform preprocessing on the face region of interest
                face = image[y:y + h, x:x + w]
                # Images_preprocessed.append(processed_face)

                x, y, s = face.shape
                if x > 0 and y > 0:
                    # Preprocess face for the FaceNet model
                    face = cv2.resize(face, (160, 160))
                    face = np.transpose(face, (2, 0, 1))
                    face = np.expand_dims(face, axis=0)
                    face = (face - 127.5) / 128

                    embeddings = resnet(torch.tensor(face).float())
                    embeddings = embeddings.detach().numpy()
                    embedding = embeddings[0].flatten()

                    _, nearest_neighbor = flann.kneighbors([embedding])
                    recognized_label = trainy[nearest_neighbor[0]]

                    # Face found in the image
            response_data = {'status': 'success', 'message': recognized_label.tolist()}
        else:
            # No face found in the image
            response_data = {'status': 'no success', 'message': 'No face found in the captured image'}

        return JsonResponse(response_data)


def prepare_images(request):
    global Face_data, Images_preprocessed, person_face, Person_data, flann, trainy

    if request.method == 'POST':
        payload = json.loads(request.body)
        name = payload.get('name', '')

        if name:
            new_row = {'Name': name, 'Image': person_face}
            Person_data = Person_data.append(new_row, ignore_index=True)

            for captures in Images_preprocessed:
                # Extract face embeddings using FaceNet
                embeddings = resnet(torch.tensor(captures).float())
                embeddings = embeddings.detach().numpy()
                embedding = embeddings[0].flatten()
                # Append a new row
                new_row = {'Name': name, 'Embeddings': embedding}
                Face_data = Face_data.append(new_row, ignore_index=True)

            Images_preprocessed = []

            # Build the ANN index using FLANN

            print('shape is : ', Face_data.shape)
            trainX = np.asarray(Face_data['Embeddings'].tolist())
            trainy = np.asarray(Face_data['Name'])
            flann.fit(trainX)

            return JsonResponse({'status': 'success', 'message': 'Data Prepared Successfully', 'Done': 'Yes'})
        else:
            return JsonResponse({'status': 'success', 'message': 'Either Name is not entered or Images are not captured', 'Done': 'No'})
    else:
        return JsonResponse({'status': 'success', 'message': 'No response'})

count = 0
l = []


def capture_images(request):
    global Images_preprocessed, face_cascade, count, person_face

    if request.method == 'POST':
        payload = json.loads(request.body)
        image_data = payload.get('image', '')

        # Convert the base64 encoded image data to OpenCV format
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if not Images_preprocessed:
            person_face = image_data

        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:

            # Process the detected faces as needed
            for (x, y, w, h) in faces:
                # Perform preprocessing on the face region of interest
                face = image[y:y + h, x:x + w]
                # Images_preprocessed.append(processed_face)

                x, y, s = face.shape
                if x > 0 and y > 0:
                    # Preprocess face for the FaceNet model
                    face = cv2.resize(face, (160, 160))
                    face = np.transpose(face, (2, 0, 1))
                    face = np.expand_dims(face, axis=0)
                    face = (face - 127.5) / 128

                    Images_preprocessed.append(face)
                    count = count + 1
                    break
                    # Face found in the image
            response_data = {'status': 'success', 'message': 'Image processed successfully', 'face_found': True}
        else:
            # No face found in the image
            response_data = {'status': 'success', 'message': 'No face found in the captured image', 'face_found': False}

        return JsonResponse(response_data)

    # Return an error response if the request method is not POST
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})


def generate_dataframe(request):
    # Pass the DataFrame to the template context
    # Convert DataFrame to a list of dictionaries
    data_list = Person_data.to_dict('records')
    print(Person_data)
    # Pass the preprocessed data to the template context
    context = {'data_list': data_list}
    # context = {'dataframe': Person_data.to_html()}
    print("Count is : ", count)
    l.append(count)
    print("List is ", l)
    print("Persons : ", len(Person_data))
    # Render the web page with the DataFrame
    return render(request, 'dataframe.html', context)
