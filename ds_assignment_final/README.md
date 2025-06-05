# WANDA Pruning 과제

## 목표
LLaMA 모델에 WANDA pruning을 적용하여 모델 크기는 줄이면서도 성능은 유지하기!!

### 과제 제출 기준
- Wikitext perplexity: 10-15 사이
- Sparsity ratio: 50% (2:4 패턴)

## 환경 설정

### 1. Conda 환경 생성 및 패키지 설치
```bash
# 1. 환경 생성 및 기본 패키지 설치
conda create -n prune_llm python=3.9 -y && \
conda activate prune_llm && \
conda install numpy=1.24.3 -y && \
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y && \
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece accelerate==0.18.0

# 2. Huggingface 패키지 업데이트
pip install -U datasets huggingface_hub
```

### 2. 디렉토리 구조
```
ds_assignment_final/
├── lib/
│   ├── prune.py        # 여기에 TODO 구현
│   ├── data.py         # 데이터 로딩 관련 코드
│   ├── eval.py         # 평가 관련 코드
│   └── ...
├── main.py             # 메인 실행 파일
├── requirements.txt
└── README.md
```

## 구현 과제

`lib/prune.py` 파일의 TODO 부분을 구현해야 합니다.

### TODO 1: Weight 초기화
```python
W = None  
```

### TODO 2: Weight Metric 계산
```python
W_metric = None  
```

### TODO 3: Mask 초기화
```python
W_mask = None  
```

### TODO 4: Pruning 조건
```python
if None != 0:  
```

### TODO 5: Block 크기 조건
```python
if None == 0:  
```

### TODO 6: WANDA Variant 함수 호출
```python
W_mask, cur_sparsity = None 
```

## 실행 방법

```bash
python main.py \
    --model baffo32/decapoda-research-llama-7B-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/wanda/
```

## 평가 방법

1. 모델이 정상적으로 pruning되었는지 확인
   - 전체 파라미터의 50%가 0이 되었는지 확인
   - 2:4 패턴이 올바르게 적용되었는지 확인

2. 성능 평가
   - Wikitext perplexity가 10-15 사이에 있는지 확인
   - 값이 낮을수록 좋음

## 참고사항

- N:M 패턴에서 N=2, M=4는 각 4개의 가중치 중 2개만 남기고 나머지는 0으로 만든다는 의미입니다.
- WANDA variant는 activation-aware pruning을 적용하는 방식입니다.
- Weight의 절대값과 activation 통계를 함께 고려하여 중요도를 계산합니다.

## 제출 방법

1. `lib/prune.py` 파일의 모든 TODO가 구현된 코드
2. 실행 결과 스크린샷 (perplexity 스코어 포함)