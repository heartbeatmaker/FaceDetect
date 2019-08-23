import cv2



#괄호안에 파일명을 쓰면 파일이 로드됌

#detecting한 얼굴을 표시할 폰트 정의

cap = cv2.VideoCapture('video.mp4') #카메라 생성

font = cv2.FONT_HERSHEY_SIMPLEX



#create the window & change the window size

#윈도우 생성 및 사이즈 변경

cv2.namedWindow('Face')



#haar 코드 사용(frontal_face) -> 어떤 파일을 쓰느냐에 따라 인식할 객체가 달라짐
'''
Haar Cascade: 머신러닝 기반의 오브젝트 검출 알고리즘
오브젝트: 얼굴 등

원하는 오브젝트를 검출하기 위해, 미리 학습시켜 놓은 xml 포맷의 분류기를 로드한다
'''

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



while(True):


# 오브젝트를 검출할 그레이스케일 이미지를 준비해 놓는다

    #read the camera image
    #카메라에서 이미지 얻기
    ret, frame = cap.read()


    #(Blue, Green, Red 계열의 이미지를 gray이미지로 변환. BGR2GRAY)
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #gray로 변환된 이미지를 cascade를 이용하여 detect

    faces = face_cascade.detectMultiScale(grayframe, 1.8, 2, 0, (30, 30))



    #얼굴을 인식하는 사각프레임에 대한 내용

    #얼굴을 인식하는 사각프레임에 넣을 글자내용

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3, 4, 0)

        cv2.putText(frame, 'Detected Face', (x-5, y-5), font, 0.9, (255,255,0),2)



    #Face로 정의된 프레임을 보여준다

    cv2.imshow('Face',frame)



    #wait keyboard input until 10ms

    #300ms 동안 키입력 대기

    #키를 누르면 꺼진다. 사진의 형태에서 얼굴 감지

    """if cv2.waitKey(300) >= 0:

		break"""



    #영상의 형태에서 얼굴 감지, space 입력시 중지

    if cv2.waitKey(1) != 255:

        break;



#close the window

#윈도우 종료

cap.release()

cv2.destroyWindow('Face')