# CV_Soccer_prediction
soccer_player_shootout_prediction  by classification image


## 페널티킥 직전 선수의 얼굴을 보고 결과를 예상하고자 하는 모델입니다. 

----
데이터는 !gdown 1-1SS2IHJnS2b9v43mFFsyXxk4SVSwLqF 로 사용 가능 

데이터 구조

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ  
|ㅡCV  
|ㅡㅡLandmark  
|ㅡㅡㅡfail  
|ㅡㅡㅡsuc  
|ㅡㅡ Landmark_ch1  
|ㅡㅡㅡgrayscale96  
|ㅡㅡㅡgrayscale160  
|ㅡㅡㅡgrayscale224  
|  
|ㅡㅡLandmark_image_sum  
|ㅡㅡㅡsum2  
|ㅡㅡㅡsum96  
|ㅡㅡㅡsum224  
|ㅡㅡ raw  
|ㅡㅡㅡfail  
|ㅡㅡㅡsuc  
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ  

**요약**

- 해당 프로젝트는 PK 직전 선수의 얼굴 사진을 딥러닝을 통해 분석하여 성공률을 예측하는 모델이다.

**역할**

- 총 3명의 팀원과 같이 기획하고, 진행하였다.
- 3개의 모델 중 VGG, Facenet 을 바탕으로 Fine-tuning 한 Image Classification을 위한 모델을 구성하고 예측 모델을 작성하였다.

**성과**

- 최종 모델을 앙상블 하여 얻은 결과는 63% acc , 를 보이고 있다.

**시기**

- 프로젝트 진행 기간 (2023. 3. 13 ~ 2023.3.30)

**목표, 모델 기대점**

- 2022 카타르 월드컵의 오프사이드 Detecting AI를 보고 스포츠와 AI의 접목이 많은 가능성을 가지고 있다고 느끼고 시작하게 되었다.
- 모델의 최종 발전 가능성은 축구에 실시간 영상을 바탕으로 한 선수의 얼굴 표정 분석으로 PK 성공률을 예측하여 시청자에게 축구 경기에 대한 흥미와 긴장감을 추가한다.



# 📝Detail

---

### 📔 데이터 set 만들기

---

- 데이터 수집
    
    기존의 데이터가 존재하지 않기에 직접 경기 영상에서 캡쳐 밑 제작. 
    
    <aside>
    💡 캡쳐는 화질이 이미지 모델에 영향을 미치기에 1080이상의 영상에서 캡쳐
    
    </aside>
    
     
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b060595c-97a6-47c4-9c14-82eb610ef2fb/Untitled.png)
    
    - 총 얻은 결과 데이터
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/267dd2bb-985f-44ca-801c-7b8506947114/Untitled.png)
    
    라벨링을 바로 직접 진행하며 시행하였다. 
    
    label ‘성공’ : 910개
    
    label ‘실패’ : 700개 
    
- 데이터 전처리
    
    기본적으로 이미지 데이터에 진행하는 전처리인 resize, gray_scaling, augmentation, zero-centering을 진행하였다. 
    
    1️⃣ Resize 
    
    사용할 모델인 VGG는 (224, 224) , Facenet은 (160, 160) , Openface는 (96, 96) 으로 resize를 
    
    진행했다. 
    
    2️⃣ Gray_scaling 
    
    기존 RGB 사진은 (224, 224, 3) 의 형식으로 구성되어 R, G, B 3채널로 이루어져 있다. 
    
    데이터 복잡도를 낮춘 데이터 셋 역시 미리 준비하였다. 
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7ddece2f-5dc7-47ee-bd31-164a5475c40f/Untitled.png)
    
    3️⃣ Augmentation
    
    수집한 data의 절대적인 숫자가 부족하기에 변경을 진행하였다. 
    
    이때 사람의 얼굴을 근간으로 진행하기에 horizon filp이 아닌 vertical filp으로 좌우 반전을 주고 , 사람의 목이 의학상으로 40도 까지 움직일 수 있기에  Rotation 역시 -40 ~ 40 안에서 변경시켰다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1b371a24-ff42-4ec2-a325-2ac902ef6b61/Untitled.png)
    
    4️⃣ zero-centering 
    
    현재 픽셀은 0~ 255 로 구성되어있어 학습 과정에서 양수로만 gradient가 움직인다. 
    
    이를 방지하고자 train_dataset의 평균으로 전체 data의 평균픽셀을 0으로 이동시켜주는 데이터를 만든다.
    
    
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12d454e5-e9e0-4b01-a3c9-377b74712a0f/Untitled.png)
    
    
## 🔎 Modeling I

---

- 모델 선정
    - VGG Face ( VGG-16)
        - 기존 VGG 모델과 마찬가지로 3 x 3 의 필터를 사용하는 특징을 가짐
        - Face data에 맞춰 학습을 시켜둔 model
        
        [Visual Geometry Group - University of Oxford](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/)
        
    - Facenet
        - 기존 이미지 분류와 다르게 face embedding을 128차원의 벡터로 나온다.
        - triplet loss ( 기준 이미지와 같은 이미지는 같게, 다른 이미지는 멀게 )  by 유클리드 거리
        
        [Papers with Code - FaceNet: A Unified Embedding for Face Recognition and Clustering](https://paperswithcode.com/paper/facenet-a-unified-embedding-for-face)
        
    - Openface
        - 기존 facenet에서 small dataset( 224 → 96 ) 으로 진행한 데이터.
        
        [OpenFace](https://cmusatyalab.github.io/openface/)
        
- 모델 빌드
    
    https://github.com/serengil/deepface_models 해당 깃허브에서 pre-train model을 load하였다. 
    
    1️⃣ VGG model 
    
    ```python
    ##VGG 7개 하위층
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)
    ```
    
    위에 아래 conv layer가 2622의 face를 분류하는 모델이기에 우리의 모델의 목표는 
    
    성공/ 실패의 두가지 경우이기에 아래에 해당하는 모델로 빌드해주었다. 
    
    기존의 얼굴 사진이라는 유사도를 가지고 있어서 fine-tune을 분류기 뿐 아니라 일부를 
    
    재학습 하는 방식을 사용했다. 
    
    2️⃣ Facenet 
    
    ```python
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 1})(up)
        x = add([x, up])
    
    # Classification block
    x = GlobalAveragePooling2D(name="AvgPool")(x)
    x = Dropout(1.0 - 0.8, name="Dropout")(x)
    # Bottleneck
    x = Dense(dimension, use_bias=False, name="Bottleneck")(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name="Bottleneck_BatchNorm")(
        x
    
    base_model_output = Sequential()
    base_model_output = Convolution2D(2, (2, 2), name='predictions')(model.layers[-5].output)
    base_model_output = Convolution2D(2, (2, 2), name='predictions1')(base_model_output)
    
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)
    ```
    
    Facenet 역시 마찬가지로 분류기 + ConV 일부분을 fine_tuning을 진행하였고 분류기를 수정된 코드 처럼 이진분류의 코드로 수정을 하였다. 
    
    3️⃣ Openface 
    
    Facenet이랑 image_size 의 차이를 제외하고는 존재하지 않기에 위와 같은 구조로 진행되었다.
    
    
    
    
    - 모델 결과 확인
    
    1️⃣ Inception Resnet ( = Facenet) 
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/69020bbf-2945-4fd2-825d-a19fe713a433/Untitled.png)
    
    다음과 같은 결과가 나온다. 특정 시점에서 과적합이 진행되는 것을 확인 할 수 있다. 현재 모델은 load해서 사용하고 있기에 이를 해소하기 위해서 데이터의 복잡도를 증가시켜야 한다고 판단하여 추가적인 절차가 필요하다고 생각했다. 
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/66d52956-56b6-4e8e-9f0f-1c1c7c5463b3/Untitled.png)
    
    accuarcy 는 73% 를 기록했고 좀 더 자세한 수치 파악을 위해 
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b3b038a8-3a3e-4ea9-92b7-8906c64873e8/Untitled.png)
    
    metrics를 통해 진행을 했다.  라벨 0 ‘실패’ 의 recall 값이 유독 낮은 것을 확인 할 수 있었다. 
    
- 추가 전처리 된 데이터 적용
    
    1️⃣ Facenet _ variation 
    
    1. augmentation : 미세한 상승이 존재하지만 눈에 띄는 상승은 X 
    2. gray_scale : 기존에 과적합이라고 생각했던 것처럼. 데이터의 복잡도를 낮추는 건 
        
        생각대로 더 낮은 결과를 보여준다. 
        
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7409defa-f039-4317-bf49-04bad8971a3e/Untitled.png)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/63719cf3-b630-46d9-94e0-1cf1e772860d/Untitled.png)
    
    수치가 recall에서는 치중되어있다.  recall에서의 수치.
    
    🌻 결 론 
    
    기존의 이미지 정확도나 F1 스코어에서 좀 더 향상된 방법론을 얻기 위한 방법이 뭐가 있을까.? 
    
    <aside>
    💡 discussion : 기존의 얼굴 인식 알고리즘이 얼굴 detecting → feature extraction → classification 이다. 여기서 기존의 68landmark 대신 3D mash feature 를 더해줘서 눈 코입, 얼굴 형의 특징을 가르쳐준다면 좀 더 좋은 결과가 나오지 않을까.
    
    </aside>
    
    
    
    






