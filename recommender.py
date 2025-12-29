import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

DATA_PATH = "dataset/"
RESULT_PATH = "test/"

def main():
    # 인자(Arguments) 처리
    if len(sys.argv) != 3:
        print("Usage: python recommender.py [training_file] [test_file]")
        sys.exit(1)

    train_file = sys.argv[1]  # 예: u1.base
    test_file = sys.argv[2]   # 예: u1.test

    print(f"Training Data: {train_file}")
    print(f"Test Data: {test_file}")

    # 데이터 로드 (Metadata 포함)
    # 컬럼 정의
    columns = ['user_id', 'item_id', 'rating', 'time_stamp']
    user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    movie_columns_base = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL']
    genre_columns = ['genre', 'genre_id']
    occupation_columns = ['occupation']

    try:
        # 학습/테스트 데이터 로드
        train_df = pd.read_csv(DATA_PATH + train_file, sep='\t', names=columns, encoding='latin-1')
        test_df = pd.read_csv(DATA_PATH + test_file, sep='\t', names=columns, encoding='latin-1')

        # 메타 데이터 로드
        user_df = pd.read_csv(DATA_PATH + 'u.user', sep='|', names=user_columns, encoding='latin-1')
        genre_df = pd.read_csv(DATA_PATH + 'u.genre', sep='|', names=genre_columns, encoding='latin-1')
        occupation_df = pd.read_csv(DATA_PATH + 'u.occupation', sep='|', names=occupation_columns, encoding='latin-1')
        
        # 영화 데이터 로드 (동적 컬럼 생성)
        genre_map = dict(zip(genre_df['genre_id'], genre_df['genre']))
        # genre_cols: ['Unknown', 'Action', 'Adventure', ... 'Western']
        genre_cols = list(genre_map.values()) 
        
        all_movie_cols = movie_columns_base + genre_cols
        movie_df = pd.read_csv(DATA_PATH + 'u.item', sep='|', names=all_movie_cols, encoding='latin-1')

    except FileNotFoundError as e:
        print(f"Error: 필수 파일을 찾을 수 없습니다. ({e})")
        print("u.user, u.item, u.genre, u.occupation 파일이 dataset 폴더에 있는지 확인해주세요.")
        sys.exit(1)

    # 데이터 전처리 및 피처 엔지니어링
    # 매핑
    user_age_map = user_df.set_index('user_id')['age']
    user_gender_map = user_df.set_index('user_id')['gender']
    user_occupation_map = user_df.set_index('user_id')['occupation']
    movie_release_map = movie_df.set_index('item_id')['release_date']

    for df in [train_df, test_df]:
        df['age'] = df['user_id'].map(user_age_map)
        df['gender'] = df['user_id'].map(user_gender_map)
        df['occupation'] = df['user_id'].map(user_occupation_map)
        df['release_date'] = df['item_id'].map(movie_release_map)

    # 장르 피처 매핑 (원-핫 인코딩된 19개 컬럼 추가)
    print("Mapping Genre features...")
    for genre in genre_cols:
        # movie_df에 있는 각 장르 컬럼(0/1)을 item_id 기준으로 가져옴
        genre_lookup = movie_df.set_index('item_id')[genre]
        train_df[genre] = train_df['item_id'].map(genre_lookup)
        test_df[genre] = test_df['item_id'].map(genre_lookup)
        # 혹시 모를 결측치는 0으로 채움
        train_df[genre] = train_df[genre].fillna(0).astype(int)
        test_df[genre] = test_df[genre].fillna(0).astype(int)

    # 날짜 처리 (Release Year)
    for df in [train_df, test_df]:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        # 결측치는 중간값으로 대체
        median_year = train_df['release_year'].median()
        df['release_year'] = df['release_year'].fillna(median_year)

    # 헬퍼 딕셔너리 생성
    item_genre_dict = {}
    item_year_dict = {}

    for index, row in movie_df.iterrows():
        item_id = row['item_id']
        item_genre_dict[item_id] = [g for g in genre_cols if row.get(g, 0) == 1]
        try:
            item_year_dict[item_id] = str(row['release_date'])[-4:]
        except:
            pass

    # 통계 피처 (User/Movie Mean)
    global_mean = train_df['rating'].mean()
    user_avg = train_df.groupby('user_id')['rating'].mean()
    movie_avg = train_df.groupby('item_id')['rating'].mean()

    train_df['user_avg_rating'] = train_df['user_id'].map(user_avg)
    train_df['movie_avg_rating'] = train_df['item_id'].map(movie_avg)
    
    test_df['user_avg_rating'] = test_df['user_id'].map(user_avg).fillna(global_mean)
    test_df['movie_avg_rating'] = test_df['item_id'].map(movie_avg).fillna(global_mean)

    # 범주형 변환
    cat_cols = ['gender', 'occupation']
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    # 모델 학습 (XGBoost)
    features = [
        'user_avg_rating', 'movie_avg_rating',
        'age', 'release_year',
        'gender', 'occupation'
    ] + genre_cols

    print(f"Training XGBoost Model with {len(features)} features...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,        # 충분히 학습
        learning_rate=0.05,       # 정교하게
        max_depth=6,              # 과적합 방지
        enable_categorical=True,
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )

    # 학습 (validation 생략)
    # 여기서는 전체 Train 데이터로 학습하여 성능을 극대화합니다.
    model.fit(
        train_df[features], 
        train_df['rating'],
        verbose=False
    )

    # 예측 및 파일 저장
    print("Predicting...")
    predictions = model.predict(test_df[features])
    
    # 예측값 범위 제한 (1~5점 사이로 클리핑)
    predictions = np.clip(predictions, 1.0, 5.0)

    # 파일명 파싱 로직
    base_name = os.path.basename(train_file) # u1.base
    output_filename = f"{base_name}_prediction.txt" # u1.base_prediction.txt
    
    print(f"Saving results to {output_filename}...")
    
    # 결과 데이터프레임 생성
    output_df = pd.DataFrame({
        'user_id': test_df['user_id'],
        'item_id': test_df['item_id'],
        'rating': predictions
    })

    # 파일 저장 (포맷: [user_id]\t[item_id]\t[rating])
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    output_df.to_csv(RESULT_PATH + output_filename, sep='\t', index=False, header=False)
    
    print("Done!")

if __name__ == "__main__":
    main()