# 프로젝트 기술 요소 전체 정리

## 프로젝트 개요

**쉐보레(Chevrolet) Manual/FAQ GraphRAG Troubleshooter**
- 목적: 쉐보레 차량 사용자 질의에 대해 PDF 매뉴얼 + FAQ 데이터 기반의 RAG 진단 시스템
- 패키지: `Chevolet_GraphRAG` (`src/Chevolet_GraphRAG/`)
- Python: >= 3.12

---

## 1. 기술 스택 (의존성)

| 카테고리 | 라이브러리 | 역할 |
|---|---|---|
| **LLM** | `langchain-openai`, `gpt-4.1-mini` | 답변 생성, Guardrail, Supervisor, 대화 이력 압축 |
| **Embedding** | `langchain-huggingface`, `BAAI/bge-m3` (1024차원) | Dense Vector 생성 (매뉴얼 + FAQ 공용) |
| **Reranker** | `langchain-cohere`, `rerank-v3.5` | 매뉴얼 페이지 + FAQ 재정렬 |
| **Graph DB** | `neo4j`, `langchain-neo4j` | 매뉴얼 계층 구조 저장 + Vector/Fulltext Index |
| **Vector DB** | `chromadb` | FAQ 벡터 저장 (PersistentClient) |
| **워크플로우** | `langgraph` | 8노드 자가수정 에이전트 DAG |
| **Observability** | `langsmith` | Trace/Latency/Cost 수집 |
| **PDF 파싱** | `pymupdf` (fitz) | 페이지 블록 추출, 이미지 렌더, 3단 레이아웃 감지 |
| **문서 변환** | `langchain-docling` | OCR Fallback |
| **Fuzzy 검색** | `rapidfuzz` | 모델명 정규화 (partial_ratio >= 78) |
| **API 서버** | `fastapi`, `uvicorn` | REST API + 정적 파일 서빙 |
| **데이터 모델** | `pydantic` | 요청/응답/내부 데이터 모델 |
| **패키지 관리** | `uv` | Python 패키지 런타임 |

---

## 2. 시스템 아키텍처 (LangGraph 8노드 워크플로우)

`src/Chevolet_GraphRAG/agent/workflow.py` 기준, 다음 8개 노드가 DAG로 연결됩니다:

```
START
  → compact_context      ← 대화 이력 압축 (8000자 초과 시 LLM 요약)
  → guardrail_check      ← 질의 전처리 + 모델 정규화 + FAQ intent 감지
  → [conditional edge]
       ↓ allow=true
  → retrieve_hybrid      ← Neo4j(벡터+키워드) + ChromaDB(FAQ) + 리랭킹
  → compose_answer       ← Source Selector + FAQ/매뉴얼 분기 답변 생성
  → supervisor_review    ← LLM 5기준 품질 검증 (PASS/REVISE)
  → evaluate_feedback    ← 피드백 평가 (부정 패턴, 저신뢰도 감지)
  → [conditional edge]
       ↓ should_requery=true
  → rewrite_query        ← 동의어 확장 + 쿼리 재작성 (최대 max_requery회)
  → retrieve_hybrid (loop)
       ↓ should_requery=false
  → finalize             ← top_manual/faq_sources 5개 제한, graph_paths 10개 제한
  → END

       ↓ allow=false (guardrail 거부)
  → finalize → END       ← "처리 불가" 응답
```

---

## 3. 데이터 저장소 이중화

### Neo4j (매뉴얼)
- 계층 구조: `Brand → Model → Manual → Section → Page → Chunk`
- 직접 연결: `Model → Chunk` (모델 필터링 검색용)
- **Vector Index** (`chunk_embedding_idx`, `image_embedding_idx`) + **Fulltext Index** (`chunk_fulltext_idx`) 동시 운용
- 시맨틱 노드: `Entity`, `Symptom`, `Action`, `DTC` + 관계 (`MENTIONS_ENTITY`, `HAS_SYMPTOM`, `RESOLVED_BY`, `REFERS_TO`)
- 순서 관계: `Chunk → NEXT_STEP → Chunk`
- 멀티모달: `Image → SUPPLEMENTS → Chunk`
- Page-centered retrieval: chunk를 검색하되 최종 출처는 page 단위로 집계
- 메타데이터: `source_file`, `page_no`, `manual_type`, `model`, `display_page_label` 보존

### ChromaDB (FAQ)
- `question + answer + category` 단순 구조
- Dense vector retrieval + Cohere rerank
- `PersistentClient`로 디스크 지속 저장: `artifacts/chroma_faq/`
- 컬렉션: `chevrolet_faq` (코사인 유사도)
- 매뉴얼 그래프를 오염시키지 않도록 완전 분리

---

## 4. 검색 파이프라인 (HybridRetriever, `hybrid.py`)

### ① 쿼리 전처리
- `_compact_query_for_embedding()`: 재질의 컨텍스트가 붙은 긴 쿼리를 임베딩용으로 압축 (최대 120자)
- `_build_fulltext_query()`: 키워드 추출 → Lucene 쿼리 빌드
- `_extract_relevance_keywords()`: 모델명/불용어(`RELEVANCE_STOPWORDS`) 제거 후 핵심 키워드 추출

### ② Dense + Lexical + FAQ 동시 검색
- `search_chunks_by_vector()`: Neo4j Vector Index에서 코사인 유사도 기반 chunk 검색 (top_k × 10, 최소 40건)
- `search_chunks_by_fulltext()`: Neo4j Fulltext Index에서 Lucene 키워드 검색 (top_k × 10, 최소 40건)
- `faq_store.search_faq()`: ChromaDB에서 FAQ vector 검색 (top_k × 3, 최소 10건)

### ③ FAQ Rerank (`_rerank_faq_hits()`)
- Cohere `rerank-v3.5` API로 FAQ 후보 재정렬
- 상위 `max(top_k × 2, 8)` 후보에 대해서만 Cross-Encoder 재정렬
- rerank_score로 FAQ 점수 갱신

### ④ Score Fusion (RRF 기반, `_fuse_manual()`)
```
vector score  = 0.62 × RRF(rank) + 0.38 × cosine_score
lexical score = 0.46 × RRF(rank) + 0.24 × normalized_lexical
preferred_manual_type bonus: vector +0.05 / lexical +0.07
```
- `excluded_chunk_ids` 자동 제외 (재질의 시 이전 결과 중복 방지)

### ⑤ Page Aggregation (`_aggregate_pages()`)
- chunk → page 그룹핑
- `page_score = top_chunk_score + support_bonus(0.06 × hits) + tail_bonus(0.03 × tail_scores) + type_bonus(0.05)`
- 같은 페이지에 chunk hit가 많을수록 가산점 (최대 3개까지)

### ⑥ Page Reranking (`_rerank_pages()`)
- Cohere `rerank-v3.5` API 호출
- 상위 `max(top_k × 3, 8)` 페이지 후보에 대해서만 Cross-Encoder 재정렬
- `rerank_score` + `relevance_label` (높음/보통/낮음) 부여

### ⑦ Relevance Pruning
- `_filter_pages_by_query_relevance()`: 키워드 overlap이 없는 page를 최종 출력에서 제거
- `_filter_faq_hits_by_query_relevance()`: 키워드 overlap이 없는 FAQ를 최종 출력에서 제거

### ⑧ Supporting Items 수집 (`_collect_supporting_items()`)
- Manual page supporting items + FAQ hits를 `prefer_faq`에 따라 우선순위 정렬
- FAQ 포함 조건: `prefer_faq=true` / Manual 결과 없음 / FAQ score >= 0.82 / (FAQ score >= 0.72 AND Manual score <= 0.45)

### ⑨ 그래프 경로 수집 (`_collect_graph_paths()`)
- 선택된 chunk_id들에서 1~2홉 시맨틱 관계 탐색
- `MENTIONS_ENTITY`, `HAS_SYMPTOM`, `RESOLVED_BY`, `REFERS_TO`, `NEXT_STEP`
- 답변 근거 및 디버그 정보로 활용

### Fallback 전략
- 모델 필터로 결과 0건 → 필터 해제 후 전체 쉐보레 데이터에서 재검색

### 선택적 확장: `graph_cypher_probe()`
- `GraphCypherQAChain`을 통한 구조화된 Cypher 질의 (현재 선택적 사용)

---

## 5. Guardrail Engine (`guardrails.py`)

단순 차단기가 아닌 **질의 전처리 계층**:

**1차 — 룰 기반 필터**:
- 타 브랜드 키워드 차단 (현대, 기아, 제네시스, 르노, 쌍용, KG, 토요타, 렉서스, 혼다, 닛산, BMW, 벤츠, 아우디, 폭스바겐, Tesla, 테슬라 — 16개)
- 비자동차 도메인 차단 (주식, 코인, 부동산, 의학, 요리, 연애 등)

**2차 — LLM-as-Judge**:
- 룰 통과 질문에 대해 GPT JSON 판정 (모델/목적 적합성)

**출력**:
- `model_candidates` 확장: `Malibu` → `["말리부", "ALL_NEW_말리부", "THE_NEW_말리부"]`
- `fallback_category` 추론: ev/sedan/suv/truck
- `preferred_manual_types` 추론: 질의 유형에 따라 `cluster_controls`/`emergency_action` 등
- `prefer_faq` (Soft Hint): FAQ intent 감지 → hard routing 없이 hint로만 사용
- `normalized_model`: 정규화된 모델명

**모델 매칭** (`_resolve_model_name`):
1. 정확 매칭 → 2. 퍼지 매칭 (`rapidfuzz.partial_ratio >= 78`) → 3. 카테고리 fallback

---

## 6. Source Selector (FAQ vs Manual 결정)

`_collect_supporting_items()` (`hybrid.py`)에서 실제 검색 점수 비교 기반으로 결정:

```python
include_top_faq = (
    prefer_faq                                            # FAQ intent hint
    or not selected_pages                                  # Manual 결과 없음
    or faq_top_score >= 0.82                              # FAQ 고점수
    or (faq_top_score >= 0.72 and manual_top_score <= 0.45)  # FAQ 우세 + Manual 약세
)
```

`_compose_answer()` (`workflow.py`)에서 최종 source 우선순위 결정:
- `prefer_faq=true` + FAQ 결과 존재 → FAQ 기반 답변 + FAQ sources 우선
- 그 외 → Manual 기반 답변 + Manual sources 우선

Hard routing 대신 **실제 검색 점수 비교** 기반의 유연한 결정

---

## 7. 답변 생성 및 품질 관리

### 프롬프트 분리
- **FAQ 모드**: "Chevrolet Customer Support FAQ Assistant" — 정책/혜택/조건/예외 설명
- **Manual 모드**: "Chevrolet Vehicle Maintenance Diagnostic Assistant" — 점검 순서/진단 절차/근거 경로 요약

### 저신뢰도 힌트
- `top_score < 0.50`이면 답변에 참고 문구 프리픽스 추가
- `"[참고: 검색된 근거의 관련도가 높지 않을 수 있습니다. 가능한 범위 내에서 답변합니다.]"`

### Supervisor Review (`_supervisor_review()`)
- 5개 기준으로 LLM이 자체 검토: intent 정합성, 환각 여부, 논리 순서, 장황함, 안전 주의사항
- `PASS` 또는 `REVISE` 결정 → REVISE 시 `---` 구분선 이후 한국어로 재작성
- 스킵 조건: 답변 없음, `confidence < 0.25`, LLM 없음

### Query Rewrite Loop (`_rewrite_query()`)
- 부정 피드백 패턴 감지 (`NEGATIVE_FEEDBACK_PATTERNS`): "해결 안됨", "여전", "동일", "아니", "실패", "still", "not fixed", "doesn't work"
- 동의어 확장 (`REWRITE_SYNONYMS`): "시동" → ["시동불량", "점화", "크랭크"] 등
- 이전 chunk_ids 제외 (`excluded_chunk_ids`, 상위 3개)로 다양성 확보
- `top_k += 2` (최대 10), `max_requery` 횟수 제한 (기본 2회)
- 재질의 형태: `{원질의} | 재질의 컨텍스트: {피드백} | 핵심키워드: {확장된 키워드(최대 12개)}`

### Requery 트리거 조건
```python
should_requery = (
    (resolved == False and negative_feedback)
    or no_evidence
    or confidence < 0.45
) and retry_count < max_requery
```

---

## 8. 데이터 적재 파이프라인

### Manual 적재 (`ingest/`)

| 모듈 | 역할 |
|---|---|
| `catalog.py` | 디렉터리/파일명에서 모델명, manual_type, 차종 카테고리 자동 추론 |
| `parser.py` | PyMuPDF 블록 기반 청킹, 3단 레이아웃 감지, OCR Fallback, Docling 보조 |
| `pipeline.py` | Neo4j 스키마 생성, 청크 임베딩 후 적재, 페이지 이미지(PNG 1.7×) 렌더, 시맨틱 엔티티 추출 |
| `schema.py` | Cypher 스키마 파일 로드 + `__EMBEDDING_DIM__` 치환 |
| `profiler.py` | 적재 통계 수집 |

**청킹 전략**: fixed-char 방식 배제 → PyMuPDF block 추출 → 짧은 블록 병합 + 긴 블록 문장 경계 분리 (420자 단위, 80자 오버랩)

**적재 필터**: `--include-models`, `--filename-keywords`, `--skip-existing`

### FAQ 적재 (`retrieval/chroma_faq.py`)

- `"Q: {question}\nA: {answer}"` 형태로 임베딩
- 동일 `BAAI/bge-m3` 임베딩 모델 사용
- 500개 단위 배치 upsert
- `--reset` 옵션으로 컬렉션 초기화

---

## 9. Score 분리 정책

| 스코어 | 용도 |
|---|---|
| `retrieval_score` | Fusion/Aggregation 내부 점수 |
| `rerank_score` | Cohere rerank 점수 (Manual page + FAQ 양쪽) |
| `relevance_label` | 사용자 표기용 (`높음/보통/낮음`) |
| `confidence` | Workflow 제어용 (`max(0.25, min(0.95, top1_score))`) |

사용자에게는 raw 점수 대신 `relevance_label`만 노출

**Confidence 활용처**:
- `< 0.25`: supervisor review 스킵
- `< 0.45`: 재질의 트리거 후보
- `< 0.50`: 답변에 저신뢰도 힌트 추가

---

## 10. 대표 이미지 정책

- Manual 페이지가 최종 source일 때만 **페이지 전체 PNG** 표시
- FAQ 선택 시(`prefer_faq=true` + FAQ 결과 존재) 이미지 없음 (unrelated 이미지로 신뢰 하락 방지)
- embedded image 사용 안 함 (텍스트와 정합성 보장 불가)
- 이미지 fallback: manual pages에 image_path가 없으면 items 전체에서 탐색

---

## 11. Observability (LangSmith)

- `tracer.trace()`: 전체 workflow를 root run으로 추적
- `tracer.event()`: confidence, retry_count, allow 등 최종 메트릭 기록
- LangChain/LangGraph와 native 연동 → 단계별 latency/cost 분석
- 환경변수: `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING`

> 참고: `langfuse_client.py`는 `LangSmithTracer`를 `LangfuseTracer` 별칭으로 re-export하는 호환 모듈

---

## 12. API 인터페이스 (`api/app.py`)

FastAPI + Uvicorn 기반 REST API (factory pattern: `create_app()`):

| 엔드포인트 | 메서드 | 설명 |
|---|---|---|
| `/` | GET | `/ui/`로 리다이렉트 |
| `/health` | GET | 헬스체크 (`status`, `database`) |
| `/chat` | POST | 질의 → 답변 + 이미지 + 매뉴얼/FAQ 출처 |
| `/feedback` | POST | 미해결 피드백 → 이전 질의로 재검색 → 재답변 |
| `/sources/top5/{session_id}` | GET | 마지막 Top-5 매뉴얼/FAQ 출처 조회 |
| `/ui` | Static | 챗봇 웹 UI (2패널) |
| `/artifacts` | Static | 페이지 이미지 정적 서빙 |

**세션 관리**: 인메모리 `SessionStore` — 스레드 안전(Lock 기반), 대화 이력/요약/디버그 관리
**이미지 경로 변환**: 절대 경로 → `/artifacts/...` 상대 경로

---

## 13. 평가 체계

### 간이 QA 평가 (`evaluate-graph`)
- JSON/TXT 질의 일괄 실행 → F1 스코어 계산

### 5-Category GraphRAG 종합 평가 (`evaluate-graphrag`)

| 카테고리 | 측정 항목 |
|---|---|
| Routing & Guardrail | 질의 검증, 모델 정규화, FAQ/매뉴얼 라우팅 정확도 |
| Graph & Document Retrieval | source_file hit@k, page hit@k, manual_type 매칭 |
| Generation & Grounding | 충실도, 답변 관련성, 사실 커버리지 |
| Multimodal & UX | 이미지-소스 정합성, 신뢰도 보정 |
| Operational Metrics | 레이턴시(p50, p95), 쿼리당 비용, 재질의 비율 |

---

## 14. CLI 명령어 (8개 서브커맨드)

```bash
# Manual 적재
uv run python -m Chevolet_GraphRAG.main ingest-data --data-root data --init-schema

# FAQ 적재
uv run python -m Chevolet_GraphRAG.main ingest-faq --faq-path data/FAQ/chevrolet_faq_target_data.json --reset

# 데이터 프로파일링
uv run python -m Chevolet_GraphRAG.main profile-data --data-root data --include-page-counts

# 단건 질의
uv run python -m Chevolet_GraphRAG.main run-graph-once --query "엔진 경고등" --model "말리부" --top-k 5

# 대화형 세션
uv run python -m Chevolet_GraphRAG.main run-graph-session --model "말리부" --top-k 5

# 간이 QA 평가
uv run python -m Chevolet_GraphRAG.main evaluate-graph --queries-file queries.json --output-file report.json

# 5-Category 종합 평가
uv run python -m Chevolet_GraphRAG.main evaluate-graphrag --dataset eval_300.json --output-file eval_report.json --use-llm

# API 서버
uv run python -m Chevolet_GraphRAG.main serve-api --host 0.0.0.0 --port 8000
```

---

## 15. 현재 구조의 한계 (명시적 TODO)

1. FAQ는 lexical search 없이 vector-only + rerank 구조
2. chunk bbox 없어 precise evidence crop 불가
3. confidence는 calibrated probability가 아닌 운영용 지표
4. GraphRAG의 `Entity/Symptom/Action/DTC` 스키마는 그래프 경로 수집에 활용되지만, retrieval ranking에는 미반영
5. `graph_cypher_probe()`는 선택적 사용 수준 — 본 검색 파이프라인에 미통합
