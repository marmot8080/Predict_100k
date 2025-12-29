# Predict_100k

**MovieLens 100k 데이터셋**을 활용하여 영화 평점을 예측하고 성능을 평가하는 프로젝트입니다.
평점 데이터와 다양한 피처(Feature) 간의 상관관계를 분석하여 최적의 조합을 찾고, 이를 통해 **RMSE(Root Mean Square Error)**를 최소화하는 것을 목표로 합니다.

## 📌 Key Features (핵심 기능)

### 1. Feature Engineering & Selection
* **상관관계 분석:** 평점(Rating)과 각 피처 간의 상관관계를 분석하여 예측에 유효한 변수를 선별합니다.
* **파생 변수 생성:**
    * **Movie Average Rating:** 해당 영화의 전체 평균 평점
    * **User Average Rating:** 해당 유저의 전체 평균 평점
* **최적 조합 탐색:** 원본 피처와 파생 변수들의 다양한 조합을 시도하여, RMSE 성능이 가장 우수한 **Best Feature Combination**을 선정하여 학습에 사용합니다.

### 2. Prediction (`recommender.py`)
* 선택된 피처 조합으로 데이터를 학습하고, 테스트 데이터셋에 대한 평점을 예측합니다.
* 예측 결과는 `test` 폴더 내에 텍스트 파일로 자동 저장됩니다.

### 3. Evaluation (`PA4.exe`)
* 채점 프로그램(`PA4.exe`)을 통해 예측된 평점 파일의 RMSE 값을 측정합니다.

---

## 🛠️ Environment (개발 환경)

* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn` (사용하신 라이브러리에 맞게 수정)

---

## 🚀 Usage (실행 방법)

### 1. 평점 예측 (Prediction)
메인 스크립트에 **학습 데이터**와 **테스트 데이터** 파일명을 인자로 주어 실행합니다.

```bash
# 사용법: python recommender.py [Train_File] [Test_File]
python recommender.py u1.base u1.test