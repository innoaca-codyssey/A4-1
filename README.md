# 지능형 문서 검색

영화 리뷰를 정제하고 NumPy TF-IDF/코사인 검색을 직접 구현합니다. Scikit-learn과 수치를 검증하고 분류/TF 변형/BM25/역색인의 결과와 BoW 한계를 분석합니다.

## 구현 요약

공개 리뷰의 식별 문자열을 입력 표현에서 제외하고 검토된 일반 단어만 사용합니다. 전처리/수식/실행 환경은 준비한 항목부터 기록합니다.


### 데이터와 전처리

Bo Pang/Lillian Lee의 Sentiment Polarity Dataset2.0을 NLTK의 고정 revision으로 확보했습니다. [배포 메타데이터](licenses/movie_reviews.xml)의 라이선스는 CC BY4.0이며 출처/해시는 resources/sources.json에 기록했습니다.

```text
{
  "source_sha256": "a41211ae685019137410268134db6a1a14428c89b671eb83056151a878539008",
  "rows": 2000,
  "positive": 1000,
  "negative": 1000,
  "dropped_zero_approved_words": 0,
  "approved_vocabulary_size": 653,
  "retained_word_fraction": 0.57641701071637,
  "data_sha256": "a0af6578a066b935490bda79fa054b9855f32af1f44ca254d633f0ccdb2fe852",
  "privacy_transform": "only reviewed generic words; other alphabetic words replaced by UNK, digits/contact syntax excluded"
}
```

긍정/부정 각각1,000개인 실제 리뷰2,000개입니다. 검토한 일반 어휘653개만 입력 표현에 남기고 그 밖의 이름 등 단어는 UNK로 대체했습니다. 숫자/연락처 형식은 제외했고 원본은 Git에 포함하지 않았습니다. 남은 어휘 비율은57.64%입니다. 알려지지 않은 단어의 위치/문장 경계를 유지해 부정어 거리도 보존했습니다. 이 제한으로 고유명사/전문 용어와 일부 의미도 함께 손실됩니다.

소문자화 후 단어 내부의 apostrophe/hyphen을 허용하고 벡터화에서 그 밖의 구두점/UNK를 제외합니다. 관사/대명사/일반 조동사는 불용어로, not/no/never와강조어는 유지합니다. movies→movie처럼 정한 명사 복수형만 통일하며 good/excellent의 강도 차이를 보존하려고 자동 동의어 통합은 하지 않습니다.

공통 어휘 중298개는 [VADER](https://github.com/cjhutto/vaderSentiment)의 일반 감성 단어를 검토해 선택했습니다. MIT 저작권은 licenses/VADER-LICENSE.txt에 보존했습니다. 해당 분석기를 호출하지 않고 단어 정제와 TF-IDF를 직접 구현합니다.

## 검증과 실험

실제 데이터와 실행 결과를 항목별로 기록합니다.

## 한계와 개선

단어 선택/문맥/어순의 영향과 실제 오분류를 검토합니다.
