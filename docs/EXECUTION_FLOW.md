# 프로젝트 실행 흐름 (Execution Flow)

> **Chevrolet Data-Centric Troubleshooting Agent** — 쉐보레 차량 취급설명서 PDF와 FAQ 데이터를 파싱·임베딩하여 Neo4j + ChromaDB에 적재하고, LangGraph 자가수정 루프를 통해 진단 답변을 생성하는 GraphRAG 에이전트의 전체 실행 흐름입니다.

---

## 목차

1. [전체 아키텍처 개요](#1-전체-아키텍처-개요)
2. [진입점 — main.py](#2-진입점--mainpy)
3. [Phase A: 매뉴얼 데이터 적재 (ingest-data)](#3-phase-a-매뉴얼-데이터-적재-ingest-data)
4. [Phase B: FAQ 데이터 적재 (ingest-faq)](#4-phase-b-faq-데이터-적재-ingest-faq)
5. [Phase C: 질의 처리 (run-graph-once / serve-api)](#5-phase-c-질의-처리-run-graph-once--serve-api)
6. [LangGraph 워크플로우 상세](#6-langgraph-워크플로우-상세)
7. [API 서버 & 웹 UI](#7-api-서버--웹-ui)
8. [보조 커맨드](#8-보조-커맨드)
9. [파일별 역할 요약표](#9-파일별-역할-요약표)

---

## 1. 전체 아키텍처 개요

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────────┐
│  PDF 매뉴얼   │ -> │  Ingest 파이프 │ -> │   Neo4j DB   │ <- │  Hybrid 검색     │
│  (data/)     │    │  라인         │    │  (GraphRAG)  │    │ (벡터+키워드+FAQ) │
└──────────────┘    └───────────────┘    └──────────────┘    └──────────────────┘
                                                                     │
┌──────────────┐    ┌───────────────┐    ┌──────────────┐            │
│  FAQ JSON    │ -> │  FAQ Ingest   │ -> │  ChromaDB    │ <──────────┘
│ (data/FAQ/)  │    │  파이프라인    │    │  (FAQ 벡터)  │
└──────────────┘    └───────────────┘    └──────────────┘
                                                │
                    ┌───────────────┐    ┌──────┴───────┐
                    │  LangGraph    │ <- │  Guardrail   │
                    │  워크플로우    │    │  Engine      │
                    │  (8노드 DAG)  │    └──────────────┘
                    └──────┬────────┘
                           │
               ┌───────────▼───────────┐
               │  FastAPI + 챗봇 UI     │
               │  (/chat, /feedback,   │
               │   /sources/top5)      │
               └───────────────────────┘
```

프로젝트는 크게 **세 단계**로 동작합니다:

| 단계 | 설명 |
|------|------|
| **Phase A** | PDF 파싱 → 텍스트/이미지 추출 → 임베딩 생성 → Neo4j 적재 |
| **Phase B** | FAQ JSON → 임베딩 생성 → ChromaDB 적재 |
| **Phase C** | 사용자 질의 → 가드레일 → 하이브리드 검색(매뉴얼+FAQ) → 답변 생성 → Supervisor 검증 → 피드백 기반 재질의 |

---

## 2. 진입점 — main.py

### 루트 `main.py`

```python
from chevy_troubleshooter.main import main
if __name__ == "__main__":
    main()
```

단순히 패키지 내부의 `main()` 함수를 호출하는 래퍼입니다.

### `src/chevy_troubleshooter/main.py`

`argparse`를 사용하여 **8개 서브커맨드**를 등록합니다:

| 커맨드 | 함수 | 설명 |
|--------|------|------|
| `ingest-data` | `cmd_ingest_data()` | PDF 파싱 → Neo4j 적재 |
| `ingest-faq` | `cmd_ingest_faq()` | FAQ JSON → ChromaDB 적재 |
| `profile-data` | `cmd_profile_data()` | 데이터 구조 프로파일링 |
| `run-graph-once` | `cmd_run_graph_once()` | 단건 질의 실행 |
| `run-graph-session` | `cmd_run_graph_session()` | 대화형 진단 세션 |
| `evaluate-graph` | `cmd_evaluate_graph()` | 간이 QA 평가 배치 실행 |
| `evaluate-graphrag` | `cmd_evaluate_graphrag()` | 5-Category GraphRAG 종합 평가 |
| `serve-api` | `cmd_serve_api()` | FastAPI 서버 구동 |

---

## 3. Phase A: 매뉴얼 데이터 적재 (ingest-data)

```bash
uv run python -m chevy_troubleshooter.main ingest-data --data-root data --init-schema
```

### Step A-1: 설정 로드 (`config.py`)

```
.env 파일 → get_settings() → Settings 데이터클래스 반환
```

- `.env`에서 Neo4j 접속 정보, LLM/Embedding 모델명, 청크 크기 등 모든 환경 변수를 로드합니다.
- `Settings` 데이터클래스에 담겨 이후 모든 컴포넌트에 전달됩니다.
- 기본값: LLM = `gpt-4.1-mini`, Embedding = `BAAI/bge-m3`, 청크 = 420자, 오버랩 = 80자

### Step A-2: PDF 탐색 (`ingest/catalog.py`)

```
data/ 폴더 재귀 탐색 → ManualFile 리스트 → DataCatalog 생성
```

- `discover_manual_files(data_root)` 함수가 `data/` 하위 차종 폴더를 순회하며 `.pdf` 파일을 모두 수집합니다.
- 각 PDF의 **상위 폴더명 = 차종명**, **파일명의 `_` 이후 부분 = 매뉴얼 유형** 으로 파싱합니다.
- 매뉴얼 유형은 `MANUAL_TYPE_MAP` 딕셔너리로 정규화합니다 (예: "계기판" → `cluster_controls`).
- 결과: `DataCatalog(manuals=[ManualFile(...), ...])` — 각 매뉴얼의 브랜드(쉐보레), 모델, 유형, 파일 경로를 포함
- 필터 옵션: `--include-models` (쉼표 구분 모델명), `--filename-keywords` (파일명 키워드), `--skip-existing` (기존 매뉴얼 스킵)

### Step A-3: PDF 파싱 (`ingest/parser.py`)

```
ManualFile → PdfManualParser.parse() → ParsedManual(pages=[PageArtifact, ...])
```

각 PDF에 대해 다음을 수행합니다:

1. **Docling 사전 추출**: `DoclingLoader`로 구조화된 청크를 먼저 추출 (선택적 보강)
2. **페이지별 처리** (PyMuPDF/fitz):
   - **3단 레이아웃 감지**: `_detect_three_column_layout()` — 텍스트 블록 중심좌표를 3등분하여 3개 컬럼 모두 활성인지 판단
   - **텍스트 추출**: 3단이면 각 컬럼별 클리핑하여 추출, 아니면 일반 추출
   - **OCR 폴백**: 텍스트가 30자 미만이면 `pytesseract` 한/영 OCR → 그래도 실패하면 3등분 크롭 후 재OCR
   - **Docling 보강**: 동일 페이지의 Docling 청크를 `[Docling]` 태그와 함께 병합
3. **텍스트 청킹**: 420자 단위, 80자 오버랩으로 슬라이딩 윈도우 분할
4. **페이지 이미지 렌더링**: 1.7배 스케일로 PNG 저장 → `artifacts/pages/{모델}/{매뉴얼}/page_XXXX.png`
5. **내장 이미지 추출**: 페이지당 최대 8개 이미지 추출 → `artifacts/embedded_images/...`

**결과물**: `ParsedManual` — `ManualFile` + 페이지별 `PageArtifact`(텍스트, 청크 리스트, 이미지 경로, 3단 여부)

### Step A-4: 임베딩 생성 (`providers.py`)

```
텍스트 청크 리스트 → build_embeddings() → SafeEmbeddings.embed_documents() → 벡터 리스트
이미지 텍스트 리스트 → 동일 임베딩 모델 → 벡터 리스트
```

- `build_embeddings(settings)` — 프로바이더 설정에 따라 `HuggingFaceEmbeddings(BAAI/bge-m3)` 또는 `OpenAIEmbeddings` 생성
- `SafeEmbeddings` 래퍼: 입력 텍스트가 `max_chars`를 초과하면 앞/뒤를 잘라 `head ... tail` 형태로 축소
- 이미지도 텍스트 기반 임베딩: `"모델:{차종} 섹션:{유형} 페이지:{번호} 이미지:{순서} {앵커텍스트}"` 형태

### Step A-5: Neo4j 스키마 초기화 (`cypher/schema.cypher`, `ingest/schema.py`)

```
schema.cypher 파일 → __EMBEDDING_DIM__ 치환 → Cypher 실행
```

12개 유니크 제약조건 + 5개 인덱스 + 2개 벡터 인덱스(chunk, image)를 생성합니다:

| 노드 타입 | 유니크 키 |
|-----------|----------|
| `Brand` | `name` |
| `Model` | `key` |
| `Manual` | `id` |
| `Section` | `id` |
| `Page` | `id` |
| `Chunk` | `id` |
| `Image` | `id` |
| `Entity` / `Symptom` / `Action` / `DTC` | `key` 또는 `code` |

벡터 인덱스: `chunk_embedding_idx`, `image_embedding_idx` (코사인 유사도, 1024차원)

### Step A-6: Neo4j 적재 (`neo4j_store.py`)

```
ParsedManual + 임베딩 벡터 → Neo4jStore.upsert_manual() → 그래프 노드/관계 생성
```

**생성되는 그래프 구조**:

```
Brand(쉐보레) -[HAS_MODEL]-> Model -[HAS_MANUAL]-> Manual -[HAS_SECTION]-> Section
                                                      ├─[HAS_PAGE]-> Page -[HAS_CHUNK]-> Chunk
                                                      │                    └─[HAS_IMAGE]-> Image -[SUPPLEMENTS]-> Chunk
                                                      └─[HAS_CHUNK]-> Chunk
Chunk -[NEXT_STEP]-> Chunk (순서 연결)
Model -[HAS_CHUNK]-> Chunk (직접 연결)
```

- 각 `Chunk` 노드에는 `embedding` 속성(1024차원 벡터)이 저장됩니다.
- 각 `Image` 노드에도 동일하게 벡터가 저장됩니다.
- `SUPPLEMENTS` 관계: 이미지 → 해당 페이지의 첫 번째 청크 연결

### Step A-7: 시맨틱 엔티티 추출 (`ingest/pipeline.py`)

```
각 청크 텍스트 → 키워드 매칭 → Entity/Symptom/Action/DTC 노드 생성 + 관계 연결
```

| 추출 대상 | 키워드 예시 | 관계 |
|-----------|------------|------|
| Entity | 엔진, 배터리, 브레이크, 변속기, 냉각수 등 12개 | `MENTIONS_ENTITY` |
| Symptom | 경고등, 시동, 소음, 진동, 과열, 누유 등 9개 | `HAS_SYMPTOM` |
| Action | 점검, 교체, 확인, 정비, 청소, 재시동, 리셋 | `RESOLVED_BY` |
| DTC 코드 | `P0420`, `B1234` 등 정규식 매칭 | `REFERS_TO` |

**Phase A 최종 결과**: JSON으로 카탈로그 요약 + 적재 통계(매뉴얼/페이지/청크/이미지 수) 출력

---

## 4. Phase B: FAQ 데이터 적재 (ingest-faq)

```bash
uv run python -m chevy_troubleshooter.main ingest-faq \
  --faq-path data/FAQ/chevrolet_faq_target_data.json --reset
```

> 파일: `src/chevy_troubleshooter/retrieval/chroma_faq.py`

### Step B-1: FAQ JSON 로드

```
data/FAQ/chevrolet_faq_target_data.json → JSON 배열 파싱
```

FAQ 데이터 구조:
```json
[
  {
    "question": "쉐보레 포인트 합산 방법은?",
    "answer": "마이쉐보레 앱에서...",
    "category": "포인트"
  }
]
```

- 각 항목을 `"Q: {question}\nA: {answer}"` 형태로 결합하여 문서화
- ID 형식: `faq::{category}::{순번}`
- 메타데이터: `category`, `question`, `source`, `source_file`

### Step B-2: FAQ 임베딩 + ChromaDB 적재

```
FAQ 문서 리스트 → build_embeddings() → 벡터 리스트 → ChromaDB upsert
```

- 동일한 `BAAI/bge-m3` 임베딩 모델 사용
- ChromaDB `PersistentClient`로 디스크 지속 저장: `artifacts/chroma_faq/`
- 컬렉션: `chevrolet_faq` (코사인 유사도)
- 500개 단위 배치 upsert
- `--reset` 옵션: 기존 컬렉션 삭제 후 재적재

**Phase B 최종 결과**: JSON으로 적재 통계(건수, 컬렉션명, 저장 경로) 출력

---

## 5. Phase C: 질의 처리 (run-graph-once / serve-api)

```bash
# 단건 질의
uv run python -m chevy_troubleshooter.main run-graph-once \
  --query "엔진 경고등이 켜지고 시동이 불안정함" --model "말리부" --top-k 5

# API 서버
uv run python -m chevy_troubleshooter.main serve-api --host 0.0.0.0 --port 8000
```

### 질의 처리 전체 흐름

```
사용자 질문
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    TroubleshootingWorkflow (8노드 DAG)                    │
│                                                                          │
│  ┌───────────────┐                                                      │
│  │compact_context│ ← 대화 이력이 길면 LLM으로 요약 (8000자 초과 시)       │
│  └──────┬────────┘                                                      │
│         ▼                                                               │
│  ┌───────────────┐     거부 → ┌──────────┐                              │
│  │guardrail_check│ ─────────> │ finalize │ → "처리 불가" 응답             │
│  └──────┬────────┘            └──────────┘                              │
│         │ 허용                                                           │
│         ▼                                                               │
│  ┌────────────────┐                                                     │
│  │retrieve_hybrid │ ← Neo4j(벡터+키워드) + ChromaDB(FAQ) + 리랭킹        │
│  └──────┬─────────┘                                                     │
│         ▼                                                               │
│  ┌────────────────┐                                                     │
│  │compose_answer  │ ← LLM으로 근거 기반 점검 절차/FAQ 답변 생성           │
│  └──────┬─────────┘                                                     │
│         ▼                                                               │
│  ┌───────────────────┐                                                  │
│  │supervisor_review  │ ← LLM으로 답변 품질 검증 (PASS/REVISE 판정)       │
│  └──────┬────────────┘                                                  │
│         ▼                                                               │
│  ┌──────────────────┐     재질의 필요 → ┌───────────────┐                │
│  │evaluate_feedback │ ──────────────> │ rewrite_query │                │
│  └──────┬───────────┘                 └──────┬────────┘                │
│         │ 종료                                │ (retrieve_hybrid        │
│         ▼                                    │  로 루프백)              │
│  ┌──────────┐ ◄──────────────────────────────┘                          │
│  │ finalize │                                                           │
│  └──────────┘                                                           │
│         │                                                               │
│         ▼                                                               │
│  최종 응답: answer + confidence + top_image_path                         │
│           + top_manual_sources + top_faq_sources                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. LangGraph 워크플로우 상세

> 파일: `src/chevy_troubleshooter/agent/workflow.py`

### 6-1. compact_context — 대화 이력 압축

- **입력**: `history_text` (이전 대화 전체)
- **처리**: `context_compaction_chars`(8000자) 초과 시 LLM에게 "진단 관련 정보만 8줄 이내 요약" 요청
- **출력**: `compact_summary` 필드에 저장 → 이후 질의에 활용

### 6-2. guardrail_check — 가드레일 검증

> 파일: `src/chevy_troubleshooter/retrieval/guardrails.py`

**2단계 혼합 가드레일**:

**1차 — 룰 기반 필터**:
- 타 브랜드 키워드 감지 (현대, 기아, 제네시스, 르노, 쌍용, KG, 토요타, 렉서스, 혼다, 닛산, BMW, 벤츠, 아우디, 폭스바겐, Tesla, 테슬라 — 16개) → 즉시 거부
- 비차량 도메인 감지 (주식, 코인, 부동산, 의학, 처방, 요리, 레시피, 다이어트, 연애, 면접, 이력서 등) → 즉시 거부

**2차 — LLM-as-Judge**:
- 룰을 통과한 질문에 대해 GPT에게 JSON 판정 요청
- 판정 포인트: 쉐보레 질의인가? / 모델이 데이터에 있는가? / 같은 차종 카테고리로 fallback 가능한가?

**모델 매칭 로직** (`_resolve_model_name`):
1. 정확 매칭: 질문에 알려진 모델명이 포함되었는지 확인
2. 퍼지 매칭: `rapidfuzz.partial_ratio >= 78` 이면 해당 모델로 매핑
3. 미매칭: `_infer_category()`로 차종 카테고리(ev/sedan/suv/truck) 추론 → 해당 카테고리의 첫 번째 모델로 fallback

**FAQ 의도 감지**:
- FAQ 관련 키워드 감지 (포인트, 오토포인트, 선포인트, faq 등) → `prefer_faq=True` 소프트 힌트
- 하드 라우팅이 아닌 점수 기반 최종 선택에 활용

**라우팅 출력**:
- `allow`: True → `retrieve_hybrid` 노드, False → `finalize` 노드 직행
- `normalized_model`: 정규화된 모델명
- `model_candidates`: 후보 모델 리스트
- `preferred_manual_types`: 매뉴얼 유형 힌트 (예: "경고등" → `cluster_controls`)
- `prefer_faq`: FAQ 우선 소프트 힌트

### 6-3. retrieve_hybrid — 하이브리드 검색

> 파일: `src/chevy_troubleshooter/retrieval/hybrid.py`

**3가지 소스에서 병렬 검색 실행**:

#### 벡터 검색 (Vector Search — Neo4j)
```
사용자 질의 → _compact_query_for_embedding() (120자 축약) → embed_query() →
Neo4j chunk_embedding_idx 벡터 인덱스 검색 → top_k×10 결과 (최소 40건)
```

- 질의 텍스트를 축약하여 임베딩 생성
- Neo4j의 벡터 인덱스로 코사인 유사도 검색

#### 키워드 검색 (Keyword Search — Neo4j)
```
사용자 질의 → _build_fulltext_query() → Neo4j fulltext 인덱스 검색 → top_k×10 결과
```

- 한/영/숫자 2글자 이상 토큰 추출, 불용어 제거
- Neo4j Lucene 기반 전문 검색 인덱스 활용

#### FAQ 검색 (Vector Search — ChromaDB)
```
사용자 질의 → embed_query() → ChromaDB 코사인 유사도 검색 → top_k×3 결과 (최소 10건)
```

- 동일 임베딩 벡터로 ChromaDB FAQ 컬렉션 검색
- 코사인 거리 → 유사도 점수 변환: `1 - (distance / 2)`

#### RRF 융합 (Reciprocal Rank Fusion — 매뉴얼만)
```
벡터 + 키워드 결과 → chunk_id 기준 합산 → RRF 점수 정렬
```

- 두 결과를 `chunk_id` 기준으로 합산하여 최종 후보 선택
- `preferred_manual_types` 적용하여 관련 매뉴얼 유형 우선
- 이전 실패 청크(`excluded_chunk_ids`)는 자동 제외

#### 페이지 집계 + 리랭킹

1. `_aggregate_pages()`: 청크를 페이지 단위로 그룹화, 점수 합산 → `PageRetrievalResult` 생성
2. `_rerank_pages()`: Cohere `rerank-v3.5` (또는 HuggingFace CrossEncoder)로 매뉴얼 페이지 리랭킹 (top_k × 2 후보)
3. `_rerank_faq_hits()`: FAQ 결과 리랭킹

#### 관련성 필터링

- `_filter_pages_by_query_relevance()`: 키워드 매칭 기반 저관련 페이지 제거
- `_filter_faq_hits_by_query_relevance()`: FAQ 동일 필터링
- `_collect_supporting_items()`: 매뉴얼 + FAQ 항목을 `prefer_faq`에 따라 우선순위 정렬

#### 그래프 경로 수집
```
선택된 chunk_id들 → MENTIONS_ENTITY/HAS_SYMPTOM/RESOLVED_BY/REFERS_TO/NEXT_STEP
1~2홉 경로 탐색 → path_summary 문자열 생성
```

**Fallback 전략**: 모델 필터로 결과가 0건이면, 필터를 해제하고 전체 쉐보레 데이터에서 재검색

### 6-4. compose_answer — 답변 생성

```
[근거1] 청크텍스트[:500]
[근거2] 청크텍스트[:500]
...
그래프 멀티홉 경로
    ↓
GPT 프롬프트 → 한국어 점검 절차 + 이유 + 근거 경로 요약
```

- 검색된 상위 5개 근거 텍스트와 그래프 경로를 LLM에게 제공
- **매뉴얼/FAQ 분기**: `prefer_faq` 여부에 따라 다른 프롬프트 사용
  - FAQ 모드: "Chevrolet Customer Support FAQ Assistant" — 혜택/조건/예외 빠짐없이 정리
  - 매뉴얼 모드: "Chevrolet Vehicle Maintenance Diagnostic Assistant" — 우선순위별 점검 절차 + 근거 경로 요약

**소스 분리 출력**:
- `top_manual_sources`: 매뉴얼 출처 (최대 7개 후보, 최종 5개)
- `top_faq_sources`: FAQ 출처 (최대 7개 후보, 최종 5개)
- `top_image_path`: 검색 결과 중 `image_path`가 있는 첫 번째 페이지 이미지 (FAQ 우선 시 생략)

**신뢰도**: `max(0.25, min(0.95, top1_score))`
- `top_score < 0.50` 시 저신뢰도 힌트 프리픽스 추가

- LLM이 없으면 룰 기반 기본 답변 템플릿 사용

### 6-5. supervisor_review — 답변 품질 검증

> 파일: `src/chevy_troubleshooter/agent/workflow.py` — `_supervisor_review()`

compose_answer 이후 LLM이 생성된 답변을 **5가지 기준**으로 검증합니다:

1. 사용자 질의 의도와의 정합성
2. 근거 외 정보 포함 여부 (할루시네이션 검사)
3. 점검 절차의 논리적 순서
4. 불필요한 장황함/중복 여부
5. 차량 안전 관련 주의사항 포함 여부

**판정**:
- `PASS`: 답변 유지
- `REVISE`: `---` 구분선 이후 수정된 한국어 답변으로 교체

**스킵 조건**: 답변 없음 또는 `confidence < 0.25` 또는 LLM 없음

### 6-6. evaluate_feedback — 피드백 평가

```python
재질의 조건 = (resolved=False AND 부정 피드백) OR 근거 없음 OR 신뢰도 < 0.45
재질의 실행 = 조건 충족 AND retry_count < max_requery(기본 2회)
```

**부정 피드백 감지 패턴**: "해결 안됨", "여전", "동일", "아니", "실패", "still", "not fixed", "doesn't work" 등

**라우팅**:
- `should_requery=True` → `rewrite_query` 노드
- `should_requery=False` → `finalize` 노드

### 6-7. rewrite_query — 질의 재구성

```
원질의 + 피드백 → 핵심 토큰 추출 + 동의어 확장 → 새 질의 생성
```

**동의어 확장 사전**:

| 키 | 확장 |
|----|------|
| 시동 | 시동불량, 점화, 크랭크 |
| 경고등 | 체크엔진, MIL, 계기판 경고 |
| 소음 | 잡음, 이상음, 떨림 |
| 브레이크 | 제동, 브레이크패드, 브레이크오일 |

**재질의 형태**:
```
{원질의} | 재질의 컨텍스트: {피드백} | 핵심키워드: {확장된 키워드들(최대 12개)}.{fallback 카테고리 텍스트}
```

**검색 전략 수정**:
- `top_k` → +2 증가 (최대 10)
- 이전 상위 3개 청크를 `excluded_chunk_ids`에 추가 (중복 방지)
- 그래프 탐색 깊이 = 2 + retry_count

→ 다시 `retrieve_hybrid` 노드로 루프백

### 6-8. finalize — 최종 처리

- `top_manual_sources`를 5개로 제한
- `top_faq_sources`를 5개로 제한
- `graph_paths`를 10개로 제한
- 기본값 보정 후 최종 상태 반환

### 관측성 — LangSmith 연동

> 파일: `src/chevy_troubleshooter/observability/langsmith_client.py`

- 워크플로우 전체를 하나의 **LangSmith trace**로 감싸서 실행합니다.
- trace 시작 시 `session_id`, `query` 를 기록하고, 종료 시 `confidence`, `retry_count`, `allow` 를 이벤트로 기록합니다.
- 환경변수: `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING`

> 참고: `langfuse_client.py`는 `LangSmithTracer`를 `LangfuseTracer` 별칭으로 re-export하는 호환 모듈입니다.

---

## 7. API 서버 & 웹 UI

### FastAPI 서버 (`api/app.py`)

```bash
uv run python -m chevy_troubleshooter.main serve-api --port 8000
```

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | `/ui/` 로 리다이렉트 |
| `/health` | GET | 서버 상태 확인 (`status`, `database`) |
| `/chat` | POST | 질의 → 답변 + 이미지 + 매뉴얼/FAQ 출처 |
| `/feedback` | POST | 미해결 피드백 → 이전 질의로 재검색 → 재답변 |
| `/sources/top5/{session_id}` | GET | 마지막 Top-5 매뉴얼/FAQ 출처 조회 |
| `/ui` | Static | 챗봇 웹 UI |
| `/artifacts` | Static | 페이지 이미지 정적 서빙 |

**요청/응답 모델**:
- `ChatRequest`: `session_id`, `user_query`, `model_hint`, `feedback`, `resolved`, `top_k`
- `ChatResponse`: `session_id`, `answer`, `confidence`, `resolved`, `top_image_path`, `top_manual_sources`, `top_faq_sources`, `top_sources`, `graph_paths`, `debug`
- `FeedbackRequest`: `session_id`, `feedback`, `resolved`
- `TopSourcesResponse`: `session_id`, `top_sources`, `top_manual_sources`, `top_faq_sources`, `top_image_path`

**세션 관리** (`agent/session_store.py`):
- 인메모리 `SessionStore`로 세션별 대화 이력/요약/디버그 정보를 관리합니다.
- 스레드 안전(Lock 기반)

**이미지 경로 변환**: 절대 경로 → `/artifacts/...` 상대 경로로 변환하여 클라이언트에 서빙

### 웹 UI (`ui_static/`)

2패널 채팅 인터페이스:

| 좌측 패널 | 우측 패널 |
|-----------|-----------|
| 채팅 로그 | Top-1 근거 페이지 이미지 |
| 질문 입력 + 모델 힌트 | Top-5 출처 목록 |

- `app.js`: `/chat` API 호출 → 답변/이미지/출처를 UI에 렌더링
- `style.css`: 쉐보레 브랜드 블루(`#0057a8`) 테마, 반응형(980px 브레이크포인트)

---

## 8. 보조 커맨드

### ingest-faq — FAQ 데이터 적재

```bash
uv run python -m chevy_troubleshooter.main ingest-faq \
  --faq-path data/FAQ/chevrolet_faq_target_data.json --reset
```

- FAQ JSON → 임베딩 → ChromaDB 적재
- `--reset`: 기존 컬렉션 삭제 후 재적재

### profile-data — 데이터 프로파일링

```bash
uv run python -m chevy_troubleshooter.main profile-data --data-root data --include-page-counts
```

- 적재 없이 데이터 구조만 파악: 모델 수, 매뉴얼 수, 유형별 분포, 모델×유형 매트릭스
- `--include-page-counts`: 각 PDF의 페이지 수까지 합산

### run-graph-session — 대화형 세션

```bash
uv run python -m chevy_troubleshooter.main run-graph-session --model 말리부 --top-k 5
```

- 터미널에서 대화형으로 질의/피드백을 반복하는 세션 모드
- 매 답변 후 피드백 입력 프롬프트 → 미해결 시 자동 재질의
- `exit` 또는 `quit`로 종료

### evaluate-graph — 간이 QA 배치 평가

```bash
uv run python -m chevy_troubleshooter.main evaluate-graph \
  --queries-file queries.json --output-file report.json --top-k 5
```

- JSON/TXT 파일 내 다수 질의를 일괄 실행
- 예상 답변과의 F1 스코어 계산 → 평가 리포트 저장
- 출력: `count`, `avg_f1`, `rows` (각 질의별 answer, f1, confidence, top_manual_sources, top_faq_sources)

### evaluate-graphrag — 5-Category GraphRAG 종합 평가

```bash
uv run python -m chevy_troubleshooter.main evaluate-graphrag \
  --dataset Comprehensive_GraphRAG_Evaluation_Dataset_300.json \
  --output-file eval_report.json --top-k 5 --use-llm
```

> 파일: `tools/evaluate_graphrag.py`

**5개 평가 카테고리**:
1. **Routing & Guardrail** — 질의 검증, 모델 정규화, FAQ/매뉴얼 라우팅 정확도
2. **Graph & Document Retrieval** — source_file hit@k, page hit@k, manual_type 매칭
3. **Generation & Grounding** — 충실도, 답변 관련성, 사실 커버리지
4. **Multimodal & UX** — 이미지-소스 정합성, 신뢰도 보정
5. **Operational Metrics** — 레이턴시(p50, p95), 쿼리당 비용, 재질의 비율

**옵션**:
- `--max-items`: 최대 평가 항목 수 제한
- `--categories`: 쉼표 구분 카테고리 필터
- `--use-llm`: LLM 기반 답변 관련성 평가 활성화

---

## 9. 파일별 역할 요약표

| 파일 경로 | 역할 |
|-----------|------|
| `main.py` (루트) | 패키지 진입점 래퍼 |
| `src/chevy_troubleshooter/main.py` | CLI 8개 서브커맨드 정의 |
| `src/chevy_troubleshooter/config.py` | 환경변수 → `Settings` 데이터클래스 |
| `src/chevy_troubleshooter/models.py` | Pydantic 데이터 모델 정의 |
| `src/chevy_troubleshooter/providers.py` | LLM/Embedding/Reranker 빌더 + `SafeEmbeddings` 래퍼 |
| `src/chevy_troubleshooter/neo4j_store.py` | Neo4j CRUD + 벡터/키워드 검색 쿼리 |
| `src/chevy_troubleshooter/cypher/schema.cypher` | Neo4j 스키마 DDL (제약조건+인덱스) |
| `src/chevy_troubleshooter/ingest/__init__.py` | ingest 패키지 공개 인터페이스 |
| `src/chevy_troubleshooter/ingest/catalog.py` | PDF 탐색 + 차종/유형 분류 |
| `src/chevy_troubleshooter/ingest/parser.py` | PDF 파싱 (PyMuPDF + Docling + OCR) |
| `src/chevy_troubleshooter/ingest/pipeline.py` | 파싱→임베딩→적재→엔티티추출 파이프라인 |
| `src/chevy_troubleshooter/ingest/profiler.py` | 데이터셋 프로파일링 유틸리티 |
| `src/chevy_troubleshooter/ingest/schema.py` | Cypher 스키마 파일 로드 유틸리티 |
| `src/chevy_troubleshooter/retrieval/__init__.py` | retrieval 패키지 공개 인터페이스 |
| `src/chevy_troubleshooter/retrieval/guardrails.py` | 룰+LLM 혼합 가드레일 + 모델 매칭 + FAQ 의도 감지 |
| `src/chevy_troubleshooter/retrieval/hybrid.py` | 벡터+키워드 하이브리드 검색 + RRF 융합 + 리랭킹 |
| `src/chevy_troubleshooter/retrieval/chroma_faq.py` | ChromaDB FAQ 벡터 저장소 (적재+검색) |
| `src/chevy_troubleshooter/agent/__init__.py` | agent 패키지 공개 인터페이스 |
| `src/chevy_troubleshooter/agent/workflow.py` | LangGraph 8노드 자가수정 워크플로우 (Supervisor 포함) |
| `src/chevy_troubleshooter/agent/session_store.py` | 인메모리 세션 대화 이력 관리 |
| `src/chevy_troubleshooter/api/app.py` | FastAPI 서버 (5 API 엔드포인트 + 정적 서빙) |
| `src/chevy_troubleshooter/observability/__init__.py` | observability 패키지 인터페이스 |
| `src/chevy_troubleshooter/observability/langsmith_client.py` | LangSmith trace/event 관측성 |
| `src/chevy_troubleshooter/observability/langfuse_client.py` | LangSmithTracer → LangfuseTracer 호환 별칭 |
| `src/chevy_troubleshooter/evaluate_retriever.py` | 검색 평가 유틸리티 |
| `tools/evaluate_graphrag.py` | 5-Category GraphRAG 종합 평가 스크립트 |
| `tools/check_neo4j.py` | Neo4j 연결 확인 유틸리티 |
| `tools/build_final_presentation.py` | 발표 자료 생성 스크립트 |
| `src/chevy_troubleshooter/ui_static/index.html` | 챗봇 UI HTML 구조 |
| `src/chevy_troubleshooter/ui_static/app.js` | 챗봇 UI 로직 (API 호출 + 렌더링) |
| `src/chevy_troubleshooter/ui_static/style.css` | 챗봇 UI 스타일 |
