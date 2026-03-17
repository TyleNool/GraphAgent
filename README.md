# Chevrolet Manual/FAQ GraphRAG

이 프로젝트는 쉐보레 차량 사용자 질의에 대해 PDF 매뉴얼과 FAQ JSON 데이터를 함께 활용하여 답변, 출처, 대표 이미지, 그리고 추적 가능한 운영 로그를 제공하는 RAG/GraphRAG 기반 진단 시스템이다.

이 README는 단순 사용법 문서가 아니라, 현재 구현된 시스템의 핵심 로직과 설계 선택의 이유를 정리한 문서다. 프로젝트 중간에 변경되었던 방향, 버린 접근, 그리고 현재 남겨둔 한계까지 포함해 "왜 지금 구조가 되었는가"를 설명한다.

## 1. 프로젝트 목표

이 프로젝트의 목표는 크게 네 가지다.

1. 사용자 질의에 대해 실제 매뉴얼/FAQ 근거 기반의 답변을 생성한다.
2. 답변뿐 아니라 출처 파일, 페이지, 대표 이미지가 사용자 질의와 자연스럽게 맞아야 한다.
3. 한국어 중심 질의, 영어/한글 혼합 키워드, 차량 세대명/별칭까지 처리해야 한다.
4. 이후 평가와 운영 분석을 위해 latency, cost, trace를 수집할 수 있어야 한다.

초기에는 "텍스트 검색이 그럴듯하면 된다"는 방향으로 시작할 수 있지만, 실제 사용 단계에서는 다음 문제가 바로 드러난다.

- 답변은 맞는데 대표 이미지가 전혀 상관없는 페이지처럼 보이는 문제
- FAQ에 거의 동일한 문장이 있는데도 PDF가 먼저 선택되는 문제
- 한국어 모델명과 영어 모델명, 세대명 표기가 달라 검색 필터가 흔들리는 문제
- 내부 score는 높아 보이는데 사용자에게 보여주는 숫자로는 의미가 다른 문제

현재 구조는 이 문제들을 해결하기 위해 여러 차례 재구성된 결과다.

## 2. 현재 시스템의 핵심 구조

현재 시스템은 다음 원칙 위에 서 있다.

- `Parent = Page`, `Child = 문단/블록 chunk`
- Manual과 FAQ는 저장소를 분리한다.
- Manual 검색과 FAQ 검색은 둘 다 수행한다.
- FAQ intent는 hard routing이 아니라 soft hint로 사용한다.
- 최종 대표 출처와 대표 이미지는 같은 결과 단위를 가리켜야 한다.
- score는 "검색 내부 점수"와 "출력용 관련도"를 분리한다.

즉, 이 프로젝트는 단순히 "벡터 검색 후 LLM 답변"이 아니라, 다음과 같은 계층 구조를 갖는 8노드 LangGraph 워크플로우 시스템이다.

1. Guardrail과 질의 해석
2. Manual / FAQ 동시 retrieval
3. Manual은 chunk → page 집계 후 rerank
4. FAQ도 별도 vector retrieval + Cohere rerank
5. 양쪽 모두 relevance pruning
6. 최종 source selector가 Manual/FAQ 중 어느 쪽이 더 맞는지 결정
7. 그 결과를 기반으로 답변 생성
8. Supervisor review로 품질 검증 후 최종 출력 (출처, 대표 이미지, trace 포함)

## 3. 왜 Neo4j와 Chroma를 같이 쓰는가

### Manual은 Neo4j

PDF 매뉴얼은 단순 문서 집합이 아니라 `모델 → 매뉴얼 → 섹션 → 페이지 → 청크`라는 구조를 가진다. 또한 최종적으로 사용자에게 보여줘야 하는 단위도 "페이지"다.

그래서 Manual 데이터는 Neo4j에 적재한다.

- `Brand → Model → Manual → Section → Page → Chunk` 구조 표현 가능
- Page와 Chunk를 동시에 조회/집계하기 쉬움
- Vector index와 fulltext index를 함께 둘 수 있음
- source file, page_no, manual_type 같은 메타데이터를 자연스럽게 유지 가능
- `Model → Chunk` 직접 연결로 모델 필터링된 검색 가능

즉 Neo4j를 선택한 이유는 "그래프를 멋있게 쓰기 위해서"가 아니라, page-centered retrieval과 metadata-aware retrieval을 가장 자연스럽게 구현하기 위해서다.

### FAQ는 Chroma

FAQ JSON은 구조가 단순하다. 대부분 `question`, `answer`, `category` 수준이면 충분하며, 페이지/섹션/그래프 관계가 필요하지 않다.

그래서 FAQ는 Chroma에 별도로 저장한다.

- 적재 구조가 단순함
- FAQ 수가 비교적 적고 독립적임 (500+ 항목)
- 빠른 vector lookup이 쉬움
- Manual graph를 불필요하게 오염시키지 않음
- 디스크 지속 저장 (`artifacts/chroma_faq/`)

이 분리는 의도적이다. FAQ까지 Neo4j에 우겨 넣으면 구조는 복잡해지고, 실제 retrieval 품질에는 큰 이득이 없다.

## 4. 데이터 적재 설계와 그 이유

### 4.1 파일 카탈로그

매뉴얼 파일은 디렉터리명과 파일명에서 모델명, 차종 카테고리, manual type을 추론한다.

- 모델명은 디렉터리/파일명 기준으로 정규화
- manual type은 파일명 키워드 기준으로 매핑 (`MANUAL_TYPE_MAP`: "계기판" → `cluster_controls`, "긴급조치" → `emergency_action` 등)
- 차종 카테고리(`ev`, `sedan`, `suv`, `truck`)는 fallback 검색용으로 사용

이 방식을 쓴 이유는 실제 데이터셋이 이미 파일명에 강한 구조적 힌트를 가지고 있기 때문이다. 따라서 질의 라우팅과 적재 메타데이터를 위해 별도 수작업 라벨링 없이도 상당한 정보를 활용할 수 있다.

적재 시 필터 옵션도 제공한다:
- `--include-models`: 특정 모델만 적재
- `--filename-keywords`: 파일명 키워드 필터
- `--skip-existing`: `source_file` 기준 중복 스킵

### 4.2 청킹: fixed-char → 블록/문단 중심

초기에는 fixed-char chunking이 단순하지만, 다음 문제가 발생한다.

- 문맥 경계가 깨짐
- 페이지 단위 근거와 연결이 약해짐
- 사용자에게 보여줄 출처 페이지와 검색 단위가 어긋남

그래서 현재는 PyMuPDF block 추출 기반으로 page 내부 블록을 읽기 순서대로 정렬하고, 짧은 블록은 합치고 긴 블록은 문장 경계 기준으로 나눈다. 기본 설정은 420자 단위, 80자 오버랩이다.

이렇게 바꾼 이유는 다음과 같다.

- 검색 정밀도는 child chunk에서 확보
- 최종 출처는 parent page로 안정적으로 표현
- 답변과 대표 이미지의 일치감을 높임

### 4.3 3단 레이아웃과 OCR fallback

매뉴얼 PDF는 일반 텍스트형 PDF만 있는 것이 아니다.

- 3단 레이아웃 문서가 존재
- 스캔형 PDF도 존재
- 페이지 번호가 인쇄되어 있지 않은 문서도 존재

그래서 현재 파서는 다음을 수행한다.

- 3단 레이아웃 감지 후 block sort order 보정
- text blocks가 너무 빈약하면 OCR fallback 수행
- OCR도 약하면 문서 변환 결과(Docling)를 보조적으로 사용
- 인쇄 페이지 번호가 있으면 `display_page_label`, 없으면 `PDF page index` 사용

이 구조의 목적은 "항상 완벽하게 예쁜 파싱"이 아니라, 실제 검색 품질에 필요한 텍스트를 최대한 안정적으로 확보하는 데 있다.

### 4.4 페이지 이미지 렌더

현재 대표 이미지로 사용되는 것은 페이지 전체 PNG다 (1.7배 스케일).

이 선택의 이유는 다음과 같다.

- PDF page를 바로 UI에 보여주기 쉽다
- 페이지 번호 기반 출처와 일관되게 연결된다
- crop/box 기반 증거 하이라이트보다 구현 리스크가 낮다

초기에 고려했던 "top chunk 영역 crop" 방식은 이론적으로 더 정확하지만, 현재 데이터에 bbox가 없고 OCR/다단 레이아웃 처리까지 필요해서 일정 대비 리스크가 컸다. 그래서 현재는 page-centered evidence display가 더 현실적이라고 판단했다.

### 4.5 embedded image를 쓰지 않는 이유

PDF에서 추출 가능한 embedded image는 존재하지만, 현재는 검색 결과 대표 이미지로 사용하지 않는다.

이유는 다음과 같다.

- 실제 사용자 질의와 직접 관련 없는 이미지가 많다
- 텍스트 근거와 이미지의 정합성을 보장하기 어렵다
- 임베딩 의미 부여가 부정확해질 가능성이 높다

결론적으로 이 프로젝트는 "가장 연관된 standalone 이미지"를 보여주기보다 "가장 연관된 페이지"를 보여주는 쪽을 선택했다.

### 4.6 FAQ 적재

FAQ 데이터는 별도 명령어(`ingest-faq`)로 ChromaDB에 적재한다.

- 각 FAQ 항목을 `"Q: {question}\nA: {answer}"` 형태로 결합하여 임베딩
- ID 형식: `faq::{category}::{순번}`
- 메타데이터: `category`, `question`, `source`, `source_file`
- 동일한 `BAAI/bge-m3` 임베딩 모델 사용
- 500개 단위 배치 upsert

FAQ를 별도 적재 단계로 분리한 이유는 Manual과 FAQ의 적재 주기가 다르기 때문이다. Manual은 신규 매뉴얼이 추가될 때만 적재하지만, FAQ는 더 자주 갱신될 수 있다.

## 5. 모델과 서비스 선택 이유

### 5.1 LLM: `gpt-4.1-mini` 기본

기본 LLM은 OpenAI `gpt-4.1-mini`이며, 설정으로 변경 가능하다 (Ollama도 지원).

선택 이유:

- Guardrail, 답변 생성, supervisor review에 필요한 언어적 안정성 확보
- 비용과 latency를 감안했을 때 실용적인 균형
- 한국어/영어 혼합 질의 대응력

이 프로젝트에서 LLM은 "모든 검색을 대신하는 모델"이 아니라, retrieval 이후의 구조화된 의사결정을 돕는 모델이다. 현재 LLM이 사용되는 곳은 다음과 같다:

1. 대화 이력 압축 (compact_context)
2. 가드레일 2차 판정 (LLM-as-Judge)
3. 답변 생성 (compose_answer) — FAQ 모드와 매뉴얼 모드 분기
4. 품질 검증 (supervisor_review) — PASS/REVISE 판정
5. LLM이 없으면 각 단계에서 룰 기반 fallback이 동작

### 5.2 Embedding: `BAAI/bge-m3`

기본 embedding 모델은 `BAAI/bge-m3` (1024차원)이다.

선택 이유:

- 한국어/영어 혼합 텍스트에 강함
- 매뉴얼, FAQ, 차량 용어, 영문 모델명/약어를 함께 다루기 유리함
- dense retrieval 품질과 범용성이 좋음

또한 Hugging Face local cache를 우선 사용하도록 구성해 (`HF_LOCAL_FILES_ONLY=true`), 오프라인/제한 네트워크 환경에서도 임베딩이 가능하도록 했다.

`SafeEmbeddings` 래퍼가 입력 텍스트 길이를 제한하여, 과도하게 긴 텍스트도 안전하게 임베딩할 수 있다.

### 5.3 Reranker: Cohere `rerank-v3.5`

최종 page rerank와 FAQ rerank 모두 Cohere reranker를 사용한다.

선택 이유:

- top candidate들 사이의 미세한 의미 차이를 잘 정리해 줌
- vector + keyword hybrid 이후의 final sorting 품질을 높임
- page 단위처럼 비교적 긴 근거 텍스트를 재정렬하기에 적절함
- HuggingFace CrossEncoder를 대체 옵션으로 지원

Reranker를 마지막 단계에만 쓰는 이유는 비용 때문이다. 모든 후보를 cross-encoder 수준으로 매번 비교하면 과도하게 무거워진다. 현재는 Manual page 후보 (top_k × 2)와 FAQ 후보 (top_k × 2, 최소 8건)에 대해서만 rerank를 수행한다.

### 5.4 Observability: LangSmith

기존 Langfuse 기반 tracing은 LangSmith로 교체했다.

선택 이유:

- LangChain/LangGraph 호출과 자연스럽게 연결됨
- trace, latency, cost 비교에 유리함
- workflow root run과 하위 child run을 한 프로젝트에서 관리하기 좋음

> 참고: `langfuse_client.py`는 `LangSmithTracer`를 `LangfuseTracer` 별칭으로 re-export하는 호환 모듈로 유지 중이다.

## 6. 검색 로직 상세

### 6.1 Guardrail

Guardrail은 단순 차단기가 아니다. 현재 역할은 다음과 같다.

- 타 브랜드/비자동차 요청 차단 (룰 기반 1차)
- LLM-as-Judge 2차 판정 (모델/목적 적합성)
- 모델명 정규화
- model candidates 확장
- fallback category 추론
- preferred manual type 추론 (예: "경고등" → `cluster_controls`)
- FAQ intent hint 추론

즉 guardrail은 retrieval 전에 "질의를 검색 가능한 형태로 정리하는 전처리 계층"이다.

### 6.2 모델명 정규화와 `model_candidates`

한국어/영어 모델명, 세대명 표기 차이 때문에 단일 문자열 필터는 잘 깨진다.

예: `Malibu` / `말리부` / `ALL_NEW_말리부` / `THE_NEW_말리부`

그래서 현재는 canonical family 개념을 내부적으로 사용하고, 실제 검색 시에는 DB 기준 후보 목록으로 확장한다.

예:

- 입력: `Malibu`
- 내부 family key: `malibu`
- 검색 후보: `["말리부", "ALL_NEW_말리부", "THE_NEW_말리부"]`

모델 매칭은 3단계로 동작한다:

1. 정확 매칭: 질문에 알려진 모델명이 포함되었는지 확인
2. 퍼지 매칭: `rapidfuzz.partial_ratio >= 78` 이면 해당 모델로 매핑
3. 미매칭: `_infer_category()`로 차종 카테고리(ev/sedan/suv/truck) 추론 → 해당 카테고리의 첫 번째 모델로 fallback

이 방식을 쓴 이유는 추가적인 DB 스키마 마이그레이션 없이도 세대명/영문/한글 차이를 안정적으로 흡수할 수 있기 때문이다.

### 6.3 Manual retrieval

Manual 쪽은 다음 순서로 검색한다.

1. dense vector search (top_k × 10, 최소 40건)
2. Neo4j fulltext search (top_k × 10, 최소 40건)
3. chunk score fusion (RRF)
4. page aggregation
5. Cohere rerank (top_k × 2 후보)
6. relevance pruning (query keyword overlap 기반)

#### Fusion 이유

정비/증상 질의는 semantic retrieval이 강하지만, 부품명/DTC/표시등/기능명은 lexical match가 강하다. 따라서 dense only도, keyword only도 부족하다.

현재 fusion은 RRF 성격과 raw score를 함께 사용하여 child chunk를 정렬한다. `preferred_manual_types`가 있으면 해당 매뉴얼 유형에 보너스를 부여한다.

#### Page aggregation 이유

검색은 child chunk 단위로 하되, 사용자에게 보여줄 최종 출처는 page 단위여야 한다. 그래서 같은 page에서 여러 hit가 나오면 support bonus를 준다.

이 선택의 이유는 다음과 같다.

- 답변과 출처 페이지를 자연스럽게 연결
- 대표 이미지와 같은 단위 유지
- 한 page 내 관련 근거가 여러 개 있는 경우를 높게 평가

### 6.4 FAQ retrieval

FAQ는 현재 Chroma 기반 vector search + Cohere rerank를 사용한다.

1. ChromaDB vector similarity search (top_k × 3, 최소 10건)
2. Cohere rerank (top_k × 2, 최소 8건 후보)
3. relevance pruning (query keyword overlap 기반)

FAQ에도 reranker를 도입한 이유:

- vector similarity만으로는 FAQ 간 미세한 의미 차이를 잡지 못하는 경우가 있었다
- 특히 유사 카테고리의 FAQ가 여러 개 있을 때, rerank 후 순서가 크게 달라지는 사례가 있었다
- Manual과 동일한 Cohere reranker를 사용하여 score scale을 맞춘다

다만 FAQ는 아직 lexical search를 별도로 수행하지 않는다. exact phrase가 강한 FAQ에 대해서는 lexical search를 추가하면 더 좋아질 수 있지만, 현재까지 누적된 실패 케이스가 그 투자를 정당화할 만큼 많지 않았다.

### 6.5 FAQ intent는 hard route가 아니라 soft hint

초기에는 FAQ intent가 잡히면 곧바로 FAQ 우선으로 가는 구조였지만, 이 방식은 지나치게 경직적이었다.

현재는 `FAQ_INTENT_HINTS` (포인트, 오토포인트, 선포인트 등)를 soft hint로만 사용한다.

즉:

- FAQ hint가 있으면 FAQ를 강하게 검토한다
- 그러나 final choice는 실제 FAQ score와 manual score를 비교해서 결정한다

이 구조의 이유:

- FAQ 키워드가 manual에도 존재할 수 있음
- hard routing은 오탐이 생기면 회복이 어려움
- 실제 검색 결과를 보고 마지막에 결정하는 것이 더 유연함

### 6.6 Source selector

현재는 Manual과 FAQ를 모두 검색한 뒤, 어느 source family가 더 적절한지 결정한다.

기본 원칙:

- FAQ hit가 강하고 FAQ hint까지 있으면 FAQ 우선
- Manual top page가 더 강하면 Manual 우선
- 애매하면 FAQ도 evidence pool에 남겨서 답변 근거로 활용

구체적인 FAQ 포함 조건 (`_collect_supporting_items`):
- `prefer_faq=True`이면 FAQ top 항목 우선 포함
- Manual 결과가 없으면 FAQ 포함
- FAQ top score >= 0.82이면 무조건 포함
- FAQ top score >= 0.72이고 Manual top score <= 0.45이면 포함

이 구조를 도입한 이유는, 실제로 "FAQ 질문인데 PDF가 먼저 나오는 문제"와 "Manual 질문인데 FAQ로 쏠리는 문제"를 동시에 줄일 수 있기 때문이다.

### 6.7 relevance pruning

`score > 0`만으로는 관련 없는 자료를 충분히 걸러내지 못한다. 그래서 retrieval 이후에 query keyword overlap 기반 relevance pruning을 추가했다.

이 단계의 목적은:

- 사용자 질의와 전혀 상관없는 page/FAQ가 최종 출력에 섞이는 현상 완화
- low-quality but positive score 결과 제거

현재 pruning은 Manual page와 FAQ 양쪽 모두에 적용된다 (`_filter_pages_by_query_relevance`, `_filter_faq_hits_by_query_relevance`). 불용어(RELEVANCE_STOPWORDS)를 제거한 키워드 기준으로 텍스트 매칭을 수행한다.

## 7. 답변 생성과 출력 정책

### 7.1 FAQ와 Manual은 다른 prompt를 사용

FAQ와 진단 매뉴얼은 답변 스타일이 다르다.

- FAQ: 혜택, 정책, 절차, 조건, 예외 설명 — "Chevrolet Customer Support FAQ Assistant"
- Manual: 점검 순서, 조치 이유, 진단 절차, 근거 경로 요약 — "Chevrolet Vehicle Maintenance Diagnostic Assistant"

그래서 source selector 결과(`prefer_faq`)에 따라 answer prompt도 분리한다.

### 7.2 Supervisor review

답변 생성 후 supervisor review가 한 번 더 품질을 확인한다.

5가지 검증 기준:
1. 사용자 질의 의도와의 정합성
2. 근거 외 정보 포함 여부 (할루시네이션)
3. 점검 절차의 논리적 순서
4. 불필요한 장황함/중복
5. 차량 안전 관련 주의사항 포함

판정:
- `PASS`: 답변 유지
- `REVISE`: `---` 구분선 이후 수정된 한국어 답변으로 교체

스킵 조건: 답변 없음, `confidence < 0.25`, LLM 없음

### 7.3 재질의 (Re-query) 메커니즘

Supervisor review 이후 `evaluate_feedback` 노드에서 재질의 필요 여부를 판단한다.

재질의 조건:
- `resolved=False` + 부정 피드백 ("해결 안됨", "여전", "동일" 등)
- 근거가 없는 경우
- `confidence < 0.45`
- 최대 재질의 횟수(`max_requery`, 기본 2회) 미달

재질의 시 질의 재구성:
- 원질의 + 피드백 결합
- 동의어 확장 (시동 → 시동불량/점화/크랭크 등)
- `top_k` +2 증가 (최대 10)
- 이전 상위 3개 청크를 `excluded_chunk_ids`에 추가

### 7.4 Confidence의 의미

현재 confidence는 "모델이 심리적으로 얼마나 확신하는가"가 아니라, top evidence score 기반의 운영용 제어값이다.

`confidence = max(0.25, min(0.95, top1_score))`

즉 confidence는 다음에 쓰인다.

- UI 표시
- supervisor skip 여부 (`< 0.25`이면 스킵)
- 재질의 트리거 (`< 0.45`이면 재질의 후보)
- 저신뢰도 힌트 (`< 0.50`이면 답변에 참고 문구 추가)

다만 score scale이 source마다 다르기 때문에, 사용자 출력에는 raw confidence보다 source 기반 `관련도 높음/보통/낮음`이 더 적합하다고 판단했다.

### 7.5 score를 왜 분리했는가

초기에는 하나의 `score`만 사용자에게 보여주었지만, Manual과 FAQ의 점수 의미가 서로 달랐다.

- Manual: fusion/page aggregation/rerank score
- FAQ: vector similarity score → rerank score

그래서 현재는 다음처럼 분리했다.

- `retrieval_score`: 검색 내부 점수
- `rerank_score`: 리랭킹 후 점수
- `relevance_label`: 사용자 출력용 (높음/보통/낮음)

이 분리의 목적은 "내부 랭킹용 숫자"와 "사용자에게 이해 가능한 출력"을 분리하는 것이다.

### 7.6 대표 이미지 정책

대표 이미지는 오직 Manual page가 최종 우선 source일 때만 보여준다.

이 선택 이유:

- FAQ는 페이지 이미지가 없음
- FAQ를 선택했는데 unrelated manual page image가 뜨면 오히려 신뢰를 해침
- 대표 이미지와 대표 출처 단위는 반드시 일치해야 함

### 7.7 출력 구조

최종 응답은 다음 필드를 포함한다:

- `answer`: 한국어 답변 텍스트
- `confidence`: 운영용 신뢰도 점수
- `top_manual_sources`: 매뉴얼 출처 (최대 5개)
- `top_faq_sources`: FAQ 출처 (최대 5개)
- `top_image_path`: 대표 페이지 이미지 경로
- `graph_paths`: 그래프 탐색 경로 (최대 10개, 디버그용)
- `debug`: 각 노드별 디버그 정보

## 8. 그래프 확장과 현재 비활성화된 부분

프로젝트는 GraphRAG 기반 확장을 염두에 두고 `Entity`, `Symptom`, `Action`, `DTC` 스키마도 일부 준비했지만, 현재 운영 경로에서는 semantic graph expansion을 적극 사용하지 않는다.

이유는 다음과 같다.

- 현재 데이터 적재 경로에서 semantic enrichment가 핵심 가치가 아니었음
- page-centered retrieval과 source alignment가 더 시급했음
- 불완전한 graph relation이 오히려 noisy warning을 발생시킴

다만 그래프 경로 수집(`_collect_graph_paths`)은 활성화되어 있어, 검색된 chunk에서 1~2홉 시맨틱 관계(MENTIONS_ENTITY, HAS_SYMPTOM, RESOLVED_BY, REFERS_TO, NEXT_STEP)를 탐색하여 답변 근거와 디버그 정보로 활용한다.

즉 현재 시스템은 "그래프를 쓴다"기보다, "그래프 구조를 수용할 수 있는 저장소 위에서 page retrieval을 우선 최적화한 상태"라고 보는 것이 맞다.

## 9. Observability와 운영 분석

LangSmith 기반 tracing은 다음 목적을 가진다.

- 질의별 workflow 추적
- 단계별 latency 측정
- provider cost 비교
- regression 전후 동작 비교

특히 이 프로젝트는 retrieval, rerank, answer generation, supervisor review가 단계적으로 이어지므로, trace 단위로 어느 단계가 병목인지 확인할 수 있어야 한다.

LangSmith를 선택한 이유는 LangChain/Graph와의 결합이 자연스럽고, cost/latency 비교 목적에 더 적합했기 때문이다.

환경변수: `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING`

## 10. 평가 체계

이 프로젝트는 단순히 answer text F1만으로 평가하기 어렵다. 현재 구조의 핵심은 다음이기 때문이다.

- 올바른 source file을 찾는가
- 올바른 page를 찾는가
- FAQ와 Manual을 올바르게 분기했는가
- 대표 이미지와 출처 페이지가 일치하는가

### 10.1 간이 QA 평가 (`evaluate-graph`)

JSON/TXT 파일 내 다수 질의를 일괄 실행하고, 예상 답변과의 F1 스코어를 계산한다.

### 10.2 5-Category GraphRAG 종합 평가 (`evaluate-graphrag`)

300개 항목 평가 데이터셋 기반으로 5개 카테고리, 15개 세부 지표를 측정한다.

| 카테고리 | 측정 항목 |
|---------|----------|
| **Routing & Guardrail** | 질의 검증, 모델 정규화, FAQ/매뉴얼 라우팅 정확도 |
| **Graph & Document Retrieval** | source_file hit@k, page hit@k, manual_type 매칭 |
| **Generation & Grounding** | 충실도, 답변 관련성, 사실 커버리지 |
| **Multimodal & UX** | 이미지-소스 정합성, 신뢰도 보정 |
| **Operational Metrics** | 레이턴시(p50, p95), 쿼리당 비용, 재질의 비율 |

즉 이 프로젝트는 "답변 문장만 비슷한가"보다 "올바른 근거를 올바른 형태로 보여주고 있는가"가 더 중요한 시스템이다.

## 11. 현재 구현 기준의 명령어

### Manual 적재

```bash
uv run python -m Chevolet_GraphRAG.main ingest-data --data-root data --init-schema
# 옵션: --max-manuals N, --include-models "모델1,모델2", --filename-keywords "키워드", --skip-existing
```

### FAQ 적재

```bash
uv run python -m Chevolet_GraphRAG.main ingest-faq --faq-path data/FAQ/chevrolet_faq_target_data.json --reset
```

### 데이터 프로파일링

```bash
uv run python -m Chevolet_GraphRAG.main profile-data --data-root data --include-page-counts
```

### 단건 질의

```bash
uv run python -m Chevolet_GraphRAG.main run-graph-once --query "엔진 경고등이 켜지고 시동이 불안정함" --model "말리부" --top-k 5
```

### 대화형 세션

```bash
uv run python -m Chevolet_GraphRAG.main run-graph-session --model "말리부" --top-k 5
```

### 간이 QA 평가

```bash
uv run python -m Chevolet_GraphRAG.main evaluate-graph --queries-file queries.json --output-file report.json --top-k 5
```

### 5-Category GraphRAG 종합 평가

```bash
uv run python -m Chevolet_GraphRAG.main evaluate-graphrag \
  --dataset Comprehensive_GraphRAG_Evaluation_Dataset_300.json \
  --output-file eval_report.json --top-k 5 --use-llm
# 옵션: --max-items N, --categories "cat1,cat2"
```

### API 실행

```bash
uv run python -m Chevolet_GraphRAG.main serve-api --host 0.0.0.0 --port 8000
# 옵션: --reload (개발 모드)
```

## 12. 현재 남아 있는 한계와 향후 우선순위

현재 구조는 운영 가능한 수준까지 정리되었지만, 아직 남아 있는 개선 여지도 분명하다.

1. FAQ는 lexical search 없이 vector-only + rerank 구조다.
2. exact phrase가 강한 FAQ는 lexical search를 붙이면 더 좋아질 수 있다.
3. chunk bbox가 없어 precise evidence crop은 아직 하지 않는다.
4. 현재 confidence는 calibrated probability가 아니라 운영용 지표다.
5. 시맨틱 그래프 확장(Entity, Symptom, Action, DTC)이 검색 품질에 미치는 영향을 아직 충분히 검증하지 않았다.

우선순위는 다음이 적절하다.

1. 실패 케이스 누적
2. FAQ lexical search 추가 여부 판단
3. 평가 지표 고도화 (5-Category 결과 기반)
4. 필요 시 evidence crop 또는 richer graph expansion 검토

## 13. 요약

이 프로젝트의 현재 버전은 다음 선택들의 결과다.

- 이미지보다 먼저 검색 품질과 source alignment를 해결한다.
- Manual은 page-centered retrieval, FAQ는 vector + rerank retrieval로 분리한다.
- FAQ intent는 soft hint만 주고, 최종 선택은 retrieval 결과를 보고 한다.
- score는 내부 랭킹용과 출력용을 분리한다.
- 대표 이미지와 대표 출처는 반드시 같은 단위를 가리키게 한다.
- 답변 생성 후 supervisor review로 품질을 검증한다.
- 운영 단계에서 latency/cost를 보기 위해 LangSmith를 사용한다.
- 5-Category 종합 평가 체계로 검색/생성/라우팅/UX/운영을 다축 평가한다.

즉 이 시스템은 "복잡한 기술을 많이 쓴 시스템"이 아니라, 실제 사용자 경험에서 가장 불편했던 문제들, 즉

- 답변은 맞는데 이미지가 틀리는 문제
- FAQ가 있는데 PDF가 먼저 나오는 문제
- 모델명 alias 때문에 검색이 흔들리는 문제

를 순서대로 해결하면서 정리된 구조다.