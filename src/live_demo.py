import cv2
import torch
from torchvision import transforms
from model import build_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

model = build_model(num_classes=len(classes))
model.load_state_dict(torch.load("../models/emotion_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w]
        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).argmax(1).item()
        label = classes[pred]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()