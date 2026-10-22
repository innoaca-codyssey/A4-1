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
  "approved_vocabulary_size": 658,
  "retained_word_fraction": 0.5769479077618698,
  "data_sha256": "a3c5e8cf98ef4940f1b2c6b7d216b38061bd785b8ee1077735254ed24c296cde",
  "privacy_transform": "only reviewed generic words; other alphabetic words replaced by UNK, digits/contact syntax excluded"
}
```

긍정/부정 각각1,000개인 실제 리뷰2,000개입니다. 검토한 일반 어휘658개만 입력 표현에 남기고 그 밖의 이름 등 단어는 UNK로 대체했습니다. 숫자/연락처 형식은 제외했고 원본은 Git에 포함하지 않았습니다. 남은 어휘 비율은57.69%입니다. 알려지지 않은 단어의 위치/문장 경계를 유지해 부정어 거리도 보존했습니다. 이 제한으로 고유명사/전문 용어와 일부 의미도 함께 손실됩니다.

소문자화 후 단어 내부의 apostrophe/hyphen을 허용하고 벡터화에서 그 밖의 구두점/UNK를 제외합니다. 관사/대명사/일반 조동사는 불용어로, not/no/never와강조어는 유지합니다. movies→movie처럼 정한 명사 복수형만 통일하며 good/excellent의 강도 차이를 보존하려고 자동 동의어 통합은 하지 않습니다.

공통 어휘 중298개는 [VADER](https://github.com/cjhutto/vaderSentiment)의 일반 감성 단어를 검토해 선택했습니다. MIT 저작권은 licenses/VADER-LICENSE.txt에 보존했습니다. 해당 분석기를 호출하지 않고 단어 정제와 TF-IDF를 직접 구현합니다.

## 검증과 실험

실제 데이터와 실행 결과를 항목별로 기록합니다.

## 한계와 개선

단어 선택/문맥/어순의 영향과 실제 오분류를 검토합니다.

### NumPy 수식과 검산

빈도는 문서별np.bincount, DF는비제로 문서 수로 계산합니다. 기본Log TF=1+ln(count), IDF=ln((N+1)/(DF+1))+1, 가중치TF*IDF를 행의L2길이로 나눕니다. 영벡터는0으로 유지하고 새 문서는 Train의 어휘/IDF로만 변환합니다. 검색에서는 두 L2벡터의 내적을 코사인으로 계산합니다.

```text
test_bm25_known_formula (test_bow.BowTests.test_bm25_known_formula) ... ok
test_bm25_zero_document_at_b_one (test_bow.BowTests.test_bm25_zero_document_at_b_one) ... ok
test_cosine_geometry (test_bow.BowTests.test_cosine_geometry) ... ok
test_double_tf_and_zero_query (test_bow.BowTests.test_double_tf_and_zero_query) ... ok
test_inverted_exact_scores (test_bow.BowTests.test_inverted_exact_scores) ... ok
test_no_identifying_string_and_negation_retained (test_bow.BowTests.test_no_identifying_string_and_negation_retained) ... ok
test_test_word_does_not_change_training_idf (test_bow.BowTests.test_test_word_does_not_change_training_idf) ... ok
test_tfidf_known_formula (test_bow.BowTests.test_tfidf_known_formula) ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.030s

OK
```

독립 손계산과 영벡터/미지어/역색인/테스트어휘누수를 검사했습니다. Scikit-learn의 TfidfVectorizer에는 같은 tokenizer/vocabulary와lowercase=False/token_pattern=None을 지정했습니다. raw/log TF, smooth_idf=True/False, norm=None/l2의8설정을 같은2,000문서에서 비교했습니다.

| TF | smooth IDF | norm | 최대 절대 오차 |
|---|---|---|---:|
| raw | True | None | 0.000e+00 |
| raw | True | l2 | 4.441e-16 |
| raw | False | None | 0.000e+00 |
| raw | False | l2 | 3.331e-16 |
| log | True | None | 0.000e+00 |
| log | True | l2 | 2.220e-16 |
| log | False | None | 0.000e+00 |
| log | False | l2 | 1.665e-16 |

모든 설정이1e-6 기준 안에 들어왔습니다. 검증에서 만든 어휘/IDF는 분류 실험에 재사용하지 않습니다.

부정 축약형 can't/ain't/needn't/mustn't/shan't도 정제에서 보존했습니다. 수정 후2,000문서의8설정을 다시 검산했고최대 오차4.44e-16, 독립8검사가 통과했습니다.
