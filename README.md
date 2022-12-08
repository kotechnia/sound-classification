
# sound-classification

❏ sound-classification의 구조는 다음과 같습니다
```bash
📂
├─ 📂data ( NIA datasets )
├─ 📂docker
├─ 📂ignite_trainer
├─ 📂model
├─ 📂protocols
│   └─ 📂nia2022
├─ 📂reproduced
├─ 📂utils
├─ 📂visdom_env
├─ 📄README.md
├─ 📄main.py
├─ 📉dataset.csv
├─ 📄prepare.py
├─ 📄requirements.txt
├─ 📉results.csv
└─ 📄test.py
```

❏ 테스트 시스템 사양은 다음과 같습니다.    
```
Ubuntu 22.04   
Python 3.8.10 
Torch 1.7.0+cu110
Torchvision==0.8.1+cu110
CUDA 11.1
cuDnn 8.2.0    
```
❏ 사용 라이브러리 및 프로그램입니다.

```
$ pip install –r requirements.txt
```

# 실행 방법

❏ dataset.csv(input csv) 생성 방법입니다.
```
python prepare.py
```
❏ dataset.csv 구조
|filename|fold|target|
|:--:|:--:|:--:|
|data/1.Training/sound_1.wav|train|0|
|data/2.Validation/sound_2.wav|val|1|
|:|:|:|
|data/3.Test/sound_n.wav|test|7|     

❏ 훈련 방법입니다.
```
python main.py --config=protocols/nia2022/nia2022.json
```


❏ 검증 방법입니다.
```
python test.py
```

# NIA 2022 sound-classification  
❏ NIA 2022 AI 학습용 데이터로 8:1:1 훈련, 검증, 실험 분할 학습 진행  
```
NIA 2022 sound-classification 데이터 총 1053h -> train 843h valid 105h test 105h  
```
※ 전체 데이터는 [AI - HUB](https://aihub.or.kr/)에서 받을 수 있습니다.  


❏ 훈련된 모델의 F1-Score 결과입니다.  
||**ESResNet**|
|:--:|:--:|
|**F1-score**|0.88|



--- 

<details>
    <summary><b>❏  original github & paper</b></summary>
    <p>github : <a href='https://github.com/AndreyGuzhov/ESResNet'>ESResNet</a>
    <p>paper : <a href='https://arxiv.org/abs/2004.07301'>arXiv:2004.07301</a>
</details>    

