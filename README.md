# 경기인력개발원 AI 수업 과제 중 했던거 설명 (현재 수정 중_)

# 요가 자세 분류 프로젝트

경량 PyTorch 기반의 실시간 요가 자세 인식 프로젝트입니다. MediaPipe 랜드마크를 이용해 자세를 추출하고, MLP 분류기로 학습·추론하며, 다양한 모드(단일 이미지, 배치, GUI, 웹 스트리밍, 헤드리스)로 사용 가능합니다.

---

## 🔍 프로젝트 개요

1. **랜드마크 추출**  
   - MediaPipe Pose 로 33개 랜드마크(x, y, z, visibility) 추출  
   - 이미지 → `landmarks/<클래스>/*.npy` (132차원 벡터)

2. **모델 학습**  
   - 간단한 MLP 분류기(PyTorch)  
   - Adam optimizer, train/val 80/20 분할, 30 에폭 학습  
   - 최적 모델 `yoga_pose_classifier.pth` 저장

3. **추론 모드**  
   - **단일 이미지**: OpenCV 오버레이 (`classify_image.py`)  
   - **배치 분류**: 폴더 전체 처리 (`classify_batch.py`)  
   - **실시간 GUI**: PySimpleGUI + Tkinter (`capture_and_classify.py`)  
   - **웹 스트리밍**: Flask + MJPEG + 캡처 API (`stream_and_capture.py`)  
   - **헤드리스 CLI**: 터미널 루프 (`headless_classify_loop.py`)

---

## 📂 데이터 구조

네, 위에서 제시된 가이드라인을 바탕으로 GitHub `README.md`에 바로 사용할 수 있도록 **마크다운(Markdown) 언어로 변환된 내용**을 작성해 드립니다. 필요한 정보를 [ ] 안에 채워 넣으시고, 스크린샷이나 GIF 등은 해당 위치에 링크나 이미지 경로를 넣어주시면 됩니다.

-----

# 🧘‍♀️ 라즈베리파이 기반 웹캠 요가 자세 분류 AI

-----

## 💡 프로젝트 소개

이 프로젝트는 **라즈베리파이에 연결된 웹캠**을 통해 사용자의 **요가 자세를 실시간으로 인식하고 분류**하는 인공지능 시스템입니다. 특정 요가 자세를 학습시킨 AI 모델을 활용하여 사용자가 올바른 자세를 취하고 있는지 판별할 수 있도록 돕습니다.

-----

## ✨ 주요 기능

  * **웹캠 기반 실시간 자세 인식**: 라즈베리파이에 연결된 웹캠을 통해 사용자의 자세를 실시간으로 캡처하고 분석합니다.
  * **요가 자세 분류**: 학습된 AI 모델을 사용하여 캡처된 자세를 미리 정의된 요가 자세(예: 다운독, 코브라, 전사 자세 등) 중 하나로 분류합니다.
  * **라즈베리파이 최적화**: 저사양 임베디드 보드인 라즈베리파이에서도 원활하게 동작하도록 모델 및 코드 최적화를 진행했습니다.

-----

## 🛠️ 개발 환경

  * **하드웨어**:
      * **라즈베리파이**: [사용하신 라즈베리파이 모델 명칭, 예: Raspberry Pi 4 Model B]
      * **웹캠**: [사용하신 웹캠 모델 명칭, 예: Logitech C920]
  * **운영체제**: Ubuntu Server [설치된 우분투 서버 버전, 예: 22.04 LTS] (라즈베리파이용)
  * **개발 도구**:
      * **VS Code (Visual Studio Code)**: SSH 원격 개발 기능을 활용하여 라즈베리파이에 연결하여 코드를 작성했습니다.
      * **SSH (Secure Shell)**: 라즈베리파이와 VS Code 간의 원격 접속 및 파일 전송에 사용했습니다.
  * **주요 라이브러리/프레임워크**:
      * [사용하신 딥러닝 프레임워크, 예: TensorFlow Lite, PyTorch 등]
      * [OpenCV (영상 처리)]
      * [NumPy (수치 계산)]
      * [기타 사용하신 파이썬 라이브러리]

-----

## 🚀 설치 및 실행 방법

### 1\. 라즈베리파이 설정

1.  라즈베리파이에 Ubuntu Server [설치된 우분투 서버 버전]를 설치합니다.
2.  SSH 접속을 위한 설정을 완료합니다.
    ```bash
    # SSH 서버 설치 (설치되어 있지 않다면)
    sudo apt update
    sudo apt install openssh-server
    ```
3.  웹캠이 라즈베리파이에 올바르게 연결되었는지 확인합니다.
    ```bash
    ls /dev/video*
    ```
    (아마도 `/dev/video0`과 같은 장치가 보여야 합니다.)

### 2\. 개발 환경 설정 (VS Code)

1.  로컬 PC에 VS Code를 설치합니다.
2.  VS Code 확장 탭에서 "**Remote - SSH**" 확장을 설치합니다.
3.  VS Code에서 `Ctrl+Shift+P` 또는 `Cmd+Shift+P`를 눌러 명령 팔레트를 열고 "**Remote-SSH: Connect to Host...**"를 선택하여 라즈베리파이에 접속합니다.

### 3\. 프로젝트 클론 및 의존성 설치

```bash
# 라즈베리파이에서
git clone [본인의 GitHub 저장소 URL]
cd [클론한 프로젝트 폴더명]
pip install -r requirements.txt
```

### 4\. 모델 학습 (선택 사항)

(만약 직접 모델을 학습시키는 코드가 있다면, 여기에 학습 방법을 간략히 설명합니다. 이미 학습된 모델을 제공한다면 이 섹션은 생략하거나 "사전 학습된 모델 사용"으로 변경하세요.)

```bash
python train_model.py
```

### 5\. 실행

```bash
python main.py # 또는 실행 스크립트 명칭
```

-----

## 📊 결과물

(여기에 프로젝트의 실행 화면 스크린샷, GIF 애니메이션, 또는 짧은 동영상 링크를 추가하면 시각적으로 프로젝트를 이해하는 데 큰 도움이 됩니다.)

  * 
  * 
  * [데모 GIF 또는 동영상 링크](https://www.google.com/search?q=https://youtu.be/your_video_link)

-----

## 🛣️ 향후 계획

  * 더 다양한 요가 자세 학습 및 분류 정확도 향상
  * 사용자 피드백 시스템 (예: 잘못된 자세 교정 가이드) 추가
  * 웹 인터페이스 또는 모바일 앱 연동
  * [생각하고 있는 추가 기능 또는 개선 사항]

-----

## 🤝 기여 방법

프로젝트 개선에 기여하고 싶으시다면 언제든지 환영합니다\!

1.  저장소를 Fork 합니다.
2.  새로운 Feature 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`).
3.  변경 사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`).
4.  브랜치에 Push 합니다 (`git push origin feature/AmazingFeature`).
5.  Pull Request를 엽니다.

-----

## 📄 라이선스

이 프로젝트는 [사용하신 라이선스, 예: MIT License]에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하십시오.

-----

## ✉️ 문의

궁금한 점이 있으시면 [본인의 GitHub 프로필 링크 또는 이메일 주소]로 연락 주십시오.

-----
