'''
동작함!!
'''


import numpy as np
import cv2
import time

'''
스크린에 떠있는 컨텐츠를 복사할 수 있는 모듈
스크린샷을 찍을 수 있다
'''
from PIL import ImageGrab


'''
Create a PNG/JPEG/GIF image object given raw data.
it will result in the image being displayed in the frontend.
'''
from IPython.display import Image



# '''
# 원하는 오브젝트를 검출하기 위해, 미리 학습시켜 놓은 xml 포맷의 분류기를 로드한다
#
# Haar Cascade: 머신러닝 기반의 오브젝트 검출 알고리즘
# 오브젝트: 얼굴 등
#
#  OpenCV에서는 이미 얼굴, 눈 등에 대한 미리 훈련된 데이터를 XML 파일 형식으로 제공한다
#  이 XML 분류자 파일을 로드 한다
#  어떤 파일을 쓰느냐에 따라 인식할 객체가 달라짐
#
#  **
#  물체 검출을 위한 자신만의 훈련이 필요하다면 OpneCV를 이용해 훈련시킬 수 있다
#  (참조 : https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html)
# '''
detector_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector_cat = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
detector_cat_ex = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')

print("detector_face="+cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
#
# cap = cv2.VideoCapture(0) #카메라 생성
#
font = cv2.FONT_HERSHEY_SIMPLEX







'''
스크린샷 데이터를 grayScale로 바꾼다
'''
def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    return processed_img



# 메인함수
def main():

    last_time = time.time()
    while True:

        # 스크린 데이터를 얻는다
        # 계속해서 스크린샷을 찍는다. 이 이미지에 오브젝트가 있는지 검사할 예정
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))

        #print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(screen)
        # cv2.imshow('window', screen)
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))



        '''스크린샷 데이터를 grayScale로 바꾼다'''
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)



        '''스크린샷 이미지에서 얼굴을 검출한다'''
        faces = detector_face.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:

            # 만약 얼굴을 발견하면, 발견한 얼굴에 대한 위치를 Rect(x,y,w,h) 형태로 얻을 수 있다
            cv2.rectangle(screen, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 얼굴을 검출하면, 'Face Detected'라는 글을 검출된 얼굴 위에 표시한다
            cv2.putText(screen, 'Face Detected', (x-5, y-5), font, 0.9, (255,255,0),2)

            # 콘솔에 메시지를 출력한다
            print("Face Detected")
            #문제점: 일단 얼굴을 검출하는 동안 이 메시지를 계속해서 띄워준다. 개별 얼굴을 구분하는 방법? 한번만 이 메시지를 띄우는 방법?



        i=0

        '''스크린샷 이미지에서 고양이를 검출한다'''
        # cats = detector_cat.detectMultiScale(gray, 1.3, 5)
        # for (x, y, w, h) in cats:
        #
        #     i+=1
        #     # 만약 얼굴을 발견하면, 발견한 얼굴에 대한 위치를 Rect(x,y,w,h) 형태로 얻을 수 있다
        #     cv2.rectangle(screen, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #
        #     # 얼굴을 검출하면, 'Face Detected'라는 글을 검출된 얼굴 위에 표시한다
        #     cv2.putText(screen, 'Cat '+format(i)+' Detected', (x-5, y-5), font, 0.9, (255,255,0),2)
        #
        #     # 콘솔에 메시지를 출력한다
        #     print("Cat Detected")
        #     #문제점: 일단 얼굴을 검출하는 동안 이 메시지를 계속해서 띄워준다. 개별 얼굴을 구분하는 방법? 한번만 이 메시지를 띄우는 방법?



        k=0
        cats_ex = detector_cat_ex.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in cats_ex:

            k+=1

            # 만약 얼굴을 발견하면, 발견한 얼굴에 대한 위치를 Rect(x,y,w,h) 형태로 얻을 수 있다
            cv2.rectangle(screen, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 얼굴을 검출하면, 'Face Detected'라는 글을 검출된 얼굴 위에 표시한다
            cv2.putText(screen, 'Cat_ex '+format(k)+' Detected', (x-5, y-5), font, 0.9, (255,255,0),2)


        # 콘솔에 메시지를 출력한다
        if k == 1:
            print(format(k)+" Cats Detected")
        if k > 1:
            print(format(k)+" Cat Detected")

        k=0


        cv2.imshow('window', screen)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()


#
# while (True):
#
#     '''
#     분류할 이미지를 Grayscale로 로드
#     '''
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
#     '''
#     입력 이미지에서 얼굴을 검출한다
#     만약 얼굴을 발견하면, 발견한 얼굴에 대한 위치를 Rect(x,y,w,h) 형태로 얻을 수 있다
#     '''
#     faces = detector.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#         # 얼굴을 검출하면, 'Face Detected'라는 글을 검출된 얼굴 위에 표시한다
#         cv2.putText(img, 'Face Detected', (x-5, y-5), font, 0.9, (255,255,0),2)
#
#         # 콘솔에 메시지를 출력한다
#         print("Face Detected") #문제점: 일단 얼굴을 검출하는 동안 이 메시지를 계속해서 띄워준다. 개별 얼굴을 구분하는 방법? 한번만 이 메시지를 띄우는 방법?
#
#
#
#     cv2.imshow('frame', img)
#
#     # break 키
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
