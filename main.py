import pandas as pd ## pandas 라이브러리를 가져오기 위하여 import를 해줍니다.
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# csv 형식으로 된 데이터 파일을 읽어옵니다.
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f'train data set은 {train.shape[1]} 개의 feature를 가진 {train.shape[0]} 개의 데이터 샘플로 이루어져 있습니다.')


print(train.shape, test.shape)

def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(train)

train['year'] = train['year'].apply(lambda x:0 if x<1900 or x>2022 else x)
test['year'] = test['year'].apply(lambda x:0 if x<1900 or x>2022 else x)

train['modelYear'] = 2022 - train['year']
test['modelYear'] = 2022 - test['year']

train['modelMeter'] = train['odometer']/train['modelYear']
test['modelMeter'] = test['odometer']/test['modelYear']

train['colorType'] = train['paint'].apply(lambda x:'chromatic' if x=='red'
                                                                  or x=='blue'
                                                                  or x=='brown'
                                                                  or x=='gold'
                                                                  or x=='green'
                                                                  or x=='orange'
                                                                  or x=='purple'
                                                                  or x=='yellow' else 'achromatic')
test['colorType'] = test['paint'].apply(lambda x:'chromatic' if x=='red'
                                                                or x=='blue'
                                                                or x=='brown'
                                                                or x=='gold'
                                                                or x=='green'
                                                                or x=='orange'
                                                                or x=='purple'
                                                                or x=='yellow' else 'achromatic')


train['brand'] = train['title'].apply(lambda x : x.split(" ")[0])
test['brand'] = test['title'].apply(lambda x:x.split(" ")[0])

train.loc[341,'target'] = 33015000
train.loc[569,'target'] = 29015000
train.loc[736,'target'] = 60015000

def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>\<]', '',
                        texts[i])  # @%*=()/+ 와 같은 문장부호 제거
        review = re.sub(r'\d+', '', review)  # 숫자 제거
        review = review.lower()  # 소문자 변환
        review = re.sub(r'\s+', ' ', review)  # extra space 제거
        review = re.sub(r'<[^>]+>', '', review)  # Html tags 제거
        review = re.sub(r'\s+', ' ', review)  # spaces 제거
        review = re.sub(r"^\s+", '', review)  # space from start 제거
        review = re.sub(r'\s+$', '', review)  # space from the end 제거
        review = re.sub(r'_', ' ', review)  # space from the end 제거
        # review = re.sub(r'l', '', review)
        corpus.append(review)

    return corpus

temp = clean_text(train['paint']) #메소드 적용
train['paint'] = temp


train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'blue' if x.find('blue') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'red' if x.find('red') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'green' if x.find('green') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'white' if x.find('white') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('grey') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('gery') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'grey' if x.find('gray') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'ash' if x.find('ash') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'brown' if x.find('brown') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'silver' if x.find('silver') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'silver' if x.find('sliver') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'black' if x.find('black') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'gold' if x.find('gold') >= 0 else x)
train['paint'] = train['paint'] = train['paint'].apply(lambda x : 'wine' if x.find('whine') >= 0 else x)

train['paint'].value_counts()


train = train.replace({
    'Abia State' : 'Abia',
    'Abuja ' : 'Abuja',
    'Lagos ' : 'Lagos',
    'Lagos State' : 'Lagos',
    'Ogun State' : 'Ogun'
    })

train = train.replace({
    'milk' : 'cream',
    'maroon' : 'red',
    'wine' : 'red',
})


brand_list = train[['brand', 'target']].groupby(['brand'], as_index=False).mean().sort_values(by='target', ascending=True,ignore_index=True)

for i, br in enumerate(brand_list.brand):
    train = train.replace({
        br: 10 * i,
    })

    test = test.replace({
        br: 10 * i,
        'Fiat': 0,
    })


train = train.replace({
    '2-cylinder(I2)' : 10,
    '3-cylinder(I3)' : 20,
    '5-cylinder(I5)' : 30,
    '4-cylinder(I4)' : 40,
    '6-cylinder(I6)' : 50,
    '6-cylinder(V6)' : 60,
    '4-cylinder(H4)' : 70,
    '8-cylinder(V8)' : 80,
    '12-cylinder(V12)' : 90,
})

test = test.replace({
    '2-cylinder(I2)' : 10,
    '3-cylinder(I3)' : 20,
    '5-cylinder(I5)' : 30,
    '4-cylinder(I4)' : 40,
    '6-cylinder(I6)' : 50,
    '6-cylinder(V6)' : 60,
    '4-cylinder(H4)' : 70,
    '8-cylinder(V8)' : 80,
    '12-cylinder(V12)' : 90,
})

train = train.replace({

    'diesel' : 10,
    'petrol' : 20,
})
test = test.replace({

    'diesel': 10,
    'petrol': 20,
})

train = train.replace({

    'manual' : 10,
    'automatic' : 20,

})
test = test.replace({

    'manual': 10,
    'automatic': 20,

})

test = test.drop('id', axis = 1) #분석에 필요없는 열 삭제
temp = clean_text(test['paint']) #메소드 적용

test['paint'] = temp
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'blue' if x.find('blue') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'red' if x.find('red') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'green' if x.find('green') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'white' if x.find('white') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('grey') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('gery') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'grey' if x.find('gray') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'ash' if x.find('ash') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'brown' if x.find('brown') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'silver' if x.find('silver') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'silver' if x.find('sliver') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'black' if x.find('black') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'gold' if x.find('gold') >= 0 else x)
test['paint'] = test['paint'] = test['paint'].apply(lambda x : 'wine' if x.find('whine') >= 0 else x)

test = test.replace({
    'Abuja ' : 'Abuja',
    'Lagos ' : 'Lagos',
    'Lagos State' : 'Lagos',
    'Ogun State' : 'Ogun',
    'Arepo ogun state ' : 'Ogun'
})

test = test.replace({
    'indigo ink pearl' : 'blue',
    'golf' : 'green',
    'maroon' : 'red',
    'wine' : 'red',
})


#라벨인코딩을 하기 위함 dictionary map 생성 함수
def make_label_map(dataframe):
    label_maps = {}
    for col in dataframe.columns:
        if dataframe[col].dtype=='object':
            label_map = {'unknown':0}
            for i, key in enumerate(dataframe[col].unique()):
                label_map[key] = i+1  #새로 등장하는 유니크 값들에 대해 1부터 1씩 증가시켜 키값을 부여해줍니다.
            label_maps[col] = label_map
    print(label_maps)
    return label_maps

# 각 범주형 변수에 인코딩 값을 부여하는 함수
def label_encoder(dataframe, label_map):
    for col in dataframe.columns:
        if dataframe[col].dtype=='object':
            dataframe[col] = dataframe[col].map(label_map[col])
            dataframe[col] = dataframe[col].fillna(label_map[col]['unknown']) #혹시 모를 결측값은 unknown의 값(0)으로 채워줍니다.
    return dataframe

le = make_label_map(train)
train = label_encoder(train, le)

X = train.drop(['id', 'target'], axis = 1) #training 데이터에서 피쳐 추출
y = train.target #training 데이터에서 중고차 가격 추출

data = train.drop('id', axis = 1).copy() #필요없는 id열 삭제
train_data, val_data = train_test_split(data, test_size=0.25) #25프로로 설정
train_data.reset_index(inplace=True) #전처리 과정에서 데이터가 뒤섞이지 않도록 인덱스를 초기화
val_data.reset_index(inplace=True)

print('학습시킬 train 셋 : ', train_data.shape)
print('검증할 val 셋 : ', val_data.shape)


train_data_X = train_data.drop(['target', 'index'], axis = 1) #training 데이터에서 피쳐 추출
train_data_y = train_data.target #training 데이터에서 target 추출

val_data_X = val_data.drop(['target', 'index'], axis = 1) #validation 데이터에서 피쳐 추출
val_data_y = val_data.target #validation 데이터에서 target 추출

#모델들을 할당할 리스트를 만들어줍니다.
models = []

#모델들을 각각 할당하여 리스트에 추가합니다.
models.append(ExtraTreesRegressor(n_estimators=110))
models.append(RandomForestRegressor(n_estimators=110))
models.append(GradientBoostingRegressor(learning_rate=0.22, criterion='friedman_mse'))

#모델들을 할당한 리스트를 불러와 순차적으로 train 데이터에 학습을 시켜줍니다.
for model in models:
    model.fit(train_data_X, train_data_y)

# 전처리가 완료된 테스트 데이터셋을 통해 본격적으로 학습한 모델로 추론을 시작합니다.
prediction = None

# 학습 된 모델들을 순차적으로 불러옵니다.
for model in models:
    # 각 모델들의 최종 회귀값들을 prediction에 모두 더해줍니다.
    if prediction is None:
        prediction = model.predict(val_data_X)
    else:
        prediction += model.predict(val_data_X)

# 앙상블에 참여한 모든 모델의 수 만큼 다시 나눠줍니다 (= 평균)
prediction /= len(models)


def nmae(true, pred):
    mae = np.mean(np.abs(true - pred))
    score = mae / np.mean(np.abs(true))

    return score


y_hat = model.predict(val_data_X)  # y예측
print(f'모델 NMAE: {nmae(val_data_y, y_hat)}')

plt.style.use('ggplot')
plt.figure(figsize=(20, 10))
plt.plot(y_hat, label = 'prediction')
plt.plot(val_data_y, label = 'real')
plt.legend(fontsize = 20)
plt.show()

train_X = train.drop(['id', 'target'], axis = 1) #training 데이터에서 피쳐 추출
train_y = train.target #training 데이터에서 target 추출

#모델들을 할당할 리스트를 만들어줍니다.
models = []

#모델들을 각각 할당하여 리스트에 추가합니다.
models.append(ExtraTreesRegressor(n_estimators=110))
models.append(RandomForestRegressor(n_estimators=110))
models.append(GradientBoostingRegressor(learning_rate=0.22, criterion='friedman_mse'))

#모델들을 할당한 리스트를 불러와 순차적으로 train 데이터에 학습을 시켜줍니다.
for model in models:
    model.fit(train_X, train_y)

check_missing_col(test) # 결측치 확인

test = label_encoder(test, le)

# 전처리가 완료된 테스트 데이터셋을 통해 본격적으로 학습한 모델로 추론을 시작합니다.
prediction = None

# 학습 된 모델들을 순차적으로 불러옵니다.
for model in models:
    # 각 모델들의 최종 회귀값들을 prediction에 모두 더해줍니다.
    if prediction is None:
        prediction = model.predict(test)
    else:
        prediction += model.predict(test)

# 앙상블에 참여한 모든 모델의 수 만큼 다시 나눠줍니다 (= 평균)
prediction /= len(models)

y_pred = model.predict(test)

# 제출용 sample 파일을 불러옵니다.
submission = pd.read_csv('data/sample_submission.csv')
submission.head()

# 위에서 구한 예측값을 그대로 넣어줍니다.
submission['target'] = y_pred

# 데이터가 잘 들어갔는지 확인합니다.
submission.head()

submission.to_csv('data/submit.csv', index=False)


