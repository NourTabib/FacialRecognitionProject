import pickle
import cv2
import numpy

def _preProcessing(face_gray):
    #Conversion vers NumpyArray
    numpy_Face = numpy.array(face_gray)
    #Redimensionner l'image
    numpy_Face = cv2.resize(numpy_Face,(128,128),interpolation = cv2.INTER_AREA)
    # Compression de l'image vers une matrice de taile (1,128*128)
    numpy_Face = numpy_Face.reshape(1,(128*128))
    return numpy_Face
def _projection_prediction(pca,cls,face) :
    numpy_Face = pca.transform(face)
    predicted = cls.predict(numpy_Face)
    return predicted

#Chargement du transformatteur
p = open('C:\\Users\\Nour.Tabib\\Desktop\\ProjetAi\\Integration\\pca.sav', 'rb')
pca = pickle.load(p)
#Chargement du Model de prediction
f = open('C:\\Users\\Nour.Tabib\\Desktop\\ProjetAi\\Integration\\classifier.sav', 'rb')
cls = pickle.load(f)
#Chargement du Model de Detection des Visages
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while(True):
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors = 5)
    for(x,y,w,h) in faces :
        face_gray = gray[y:y+h,x:x+w]
        face_colored = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        numpy_Face = _preProcessing(face_gray)
        predicted = _projection_prediction(pca,cls,numpy_Face)
        cv2.putText(frame,predicted[0],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()