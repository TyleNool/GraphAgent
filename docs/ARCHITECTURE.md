# Architecture Notes

## 1) Neo4j 스키마 (요약)

노드:
- `Brand`, `Model`, `Manual`, `Section`, `Page`, `Chunk`, `Image`
- 시맨틱 노드: `Entity`, `Symptom`, `Action`, `DTC`

관계:
- 구조: `HAS_MODEL`, `HAS_MANUAL`, `HAS_SECTION`, `HAS_PAGE`, `HAS_CHUNK`, `HAS_IMAGE`
- 직접 연결: `Model -[HAS_CHUNK]-> Chunk`
- 시맨틱: `MENTIONS_ENTITY`, `HAS_SYMPTOM`, `RESOLVED_BY`, `REFERS_TO`
- 순서: `NEXT_STEP` (Chunk → Chunk 순서 연결)
- 멀티모달: `SUPPLEMENTS` (Image → 해당 페이지 첫 번째 Chunk)

주요 provenance 필드:
- `source_file`, `page_no`, `image_path`, `updated_at`

그래프 구조:
```
Brand(쉐보레) -[HAS_MODEL]-> Model -[HAS_MANUAL]-> Manual -[HAS_SECTION]-> Section
                                                     ├─[HAS_PAGE]-> Page -[HAS_CHUNK]-> Chunk
                                                     │                    └─[HAS_IMAGE]-> Image -[SUPPLEMENTS]-> Chunk
                                                     └─[HAS_CHUNK]-> Chunk
Model -[HAS_CHUNK]-> Chunk (직접 연결)
Chunk -[NEXT_STEP]-> Chunk (순서 연결)
```

유니크 제약조건: Brand(`name`), Model(`key`), Manual(`id`), Section(`id`), Page(`id`), Chunk(`id`), Image(`id`), Entity/Symptom/Action(`key`), DTC(`code`)

벡터 인덱스: `chunk_embedding_idx`, `image_embedding_idx` (코사인 유사도, 1024차원 — BAAI/bge-m3)

전문 검색 인덱스: `chunk_fulltext_idx` (Lucene)

실제 Cypher는 `src/Chevolet_GraphRAG/cypher/schema.cypher` 참조.

## 2) LangGraph 워크플로우 (8노드 자가수정 루프)

노드:
- `compact_context` — 대화 이력이 길면 LLM으로 요약 (8000자 초과 시)
- `guardrail_check` — 2단계 혼합 가드레일 (룰 기반 + LLM-as-Judge)
- `retrieve_hybrid` — Neo4j 벡터+키워드 검색 + ChromaDB FAQ 검색 + RRF 융합 + 리랭킹
- `compose_answer` — LLM 기반 근거 기반 점검 절차 생성
- `supervisor_review` — LLM 기반 품질 검증 (PASS/REVISE 판정)
- `evaluate_feedback` — 재질의 필요 여부 판단
- `rewrite_query` — 동의어 확장 + 피드백 통합으로 질의 재구성
- `finalize` — 결과 정리 및 top-k 제한

엣지:
```
START → compact_context → guardrail_check
  ├─[allow=true]→ retrieve_hybrid → compose_answer → supervisor_review → evaluate_feedback
  │                                                                        ├─[should_requery=true]→ rewrite_query → retrieve_hybrid (루프백)
  │                                                                        └─[should_requery=false]→ finalize → END
  └─[allow=false]→ finalize → END
```

## 3) Guardrail 혼합 전략

> 파일: `src/Chevolet_GraphRAG/retrieval/guardrails.py`

1차 룰 기반:
- 타 브랜드 키워드 포함 시 거부 (현대, 기아, 제네시스, 르노, 쌍용, KG, 토요타, 렉서스, 혼다, 닛산, BMW, 벤츠, 아우디, 폭스바겐, Tesla, 테슬라)
- 비차량 도메인 키워드 포함 시 거부 (주식, 코인, 부동산, 의학, 요리, 연애 등)

2차 LLM-as-Judge:
- 룰 통과 요청에 대해 모델/목적 적합성 판단
- 데이터 미존재 모델은 쉐보레 동일 차종 카테고리 fallback 허용

모델 매칭 로직 (`_resolve_model_name`):
1. 정확 매칭: 질문에 알려진 모델명이 포함되었는지 확인
2. 퍼지 매칭: `rapidfuzz.partial_ratio >= 78` 이면 해당 모델로 매핑
3. 미매칭 시: `_infer_category()`로 차종 카테고리(ev/sedan/suv/truck) 추론 → 해당 카테고리 fallback

라우팅 결정:
- `allow=True` → `retrieve_hybrid` 노드로 진행
- `allow=False` → `finalize` 노드로 직행 (거부 사유 포함 응답)

추가 출력:
- `normalized_model`: 정규화된 모델명
- `model_candidates`: 해당 카테고리 후보 모델 리스트
- `preferred_manual_types`: 질의에 적합한 매뉴얼 유형 힌트 (예: "경고등" → `cluster_controls`)
- `prefer_faq`: FAQ 우선 라우팅 소프트 힌트 (하드 라우팅 아님)

## 4) 하이브리드 검색 (Hybrid Retrieval)

> 파일: `src/Chevolet_GraphRAG/retrieval/hybrid.py`

### 검색 파이프라인

1. **질의 전처리**:
   - `_compact_query_for_embedding()`: 120자로 축약하여 임베딩 생성
   - `_build_fulltext_query()`: 키워드 추출 → Lucene 쿼리 생성

2. **병렬 검색** (3가지 소스):
   - Neo4j 벡터 검색: `search_chunks_by_vector()` (top_k × 10)
   - Neo4j 키워드 검색: `search_chunks_by_fulltext()` (top_k × 10)
   - ChromaDB FAQ 벡터 검색: `faq_store.search_faq()` (top_k × 3)

3. **RRF 점수 융합** (`_fuse_manual()`):
   - Reciprocal Rank Fusion으로 벡터 + 키워드 결과 합산
   - `chunk_id` 기준 중복 제거
   - `manual_type` 선호도 필터링 적용

4. **페이지 집계** (`_aggregate_pages()`):
   - 청크를 페이지 단위로 그룹화, 점수 합산
   - `PageRetrievalResult` 생성 (supporting_items 포함)

5. **리랭킹**:
   - `_rerank_pages()`: Cohere/HuggingFace CrossEncoder로 매뉴얼 결과 리랭킹 (top_k × 2 후보)
   - `_rerank_faq_hits()`: FAQ 결과 리랭킹

6. **관련성 필터링**:
   - `_filter_pages_by_query_relevance()`: 키워드 매칭 기반 저관련 페이지 제거
   - `_filter_faq_hits_by_query_relevance()`: FAQ 동일 필터링

### Fallback 전략
- 모델 필터 결과 0건 → 필터 해제 후 전체 쉐보레 데이터에서 재검색
- 이전 실패 청크 자동 제외 (`excluded_chunk_ids`)

## 5) Supervisor Review (품질 검증)

> 파일: `src/Chevolet_GraphRAG/agent/workflow.py` — `_supervisor_review()`

compose_answer 이후 LLM이 생성된 답변을 5가지 기준으로 검증:
1. 사용자 질의 의도와의 정합성
2. 근거 외 정보 포함 여부 (할루시네이션 검사)
3. 점검 절차의 논리적 순서
4. 불필요한 장황함/중복 여부
5. 차량 안전 관련 주의사항 포함 여부

판정:
- `PASS`: 답변 유지
- `REVISE`: 수정된 답변으로 교체 (한국어 고정)
- 답변 없음 또는 confidence < 0.25: 스킵

## 6) Re-query 세부 알고리즘

입력:
- 사용자 피드백, 해결 여부(`resolved`), 현재 confidence, 기존 검색 근거

규칙:
1. `resolved=True`면 즉시 종료
2. `resolved=False` + 부정 피드백 패턴(예: "해결 안됨", "여전", "동일", "아니", "still", "not fixed")이면 재질의
3. 근거 없음/신뢰도 < 0.45이면 재질의
4. 반복 상한(`max_requery`, 기본 2회) 도달 시 종료

재질의 생성:
- 원질의 + 피드백 결합
- 핵심어 추출(한글/영문/숫자 토큰)
- 증상 동의어 확장:

| 키 | 확장 |
|----|------|
| 시동 | 시동불량, 점화, 크랭크 |
| 경고등 | 체크엔진, MIL, 계기판 경고 |
| 소음 | 잡음, 이상음, 떨림 |
| 브레이크 | 제동, 브레이크패드, 브레이크오일 |

- fallback 카테고리 안내 문구 추가
- 재질의 형태: `{원질의} | 재질의 컨텍스트: {피드백} | 핵심키워드: {확장된 키워드들}. 동급 {카테고리} 차종 근거 우선 검색`

검색 전략 재설정:
- `top_k` → +2 증가 (최대 10)
- 이전 상위 3개 청크를 `excluded_chunk_ids`에 추가 (중복 방지)
- 그래프 탐색 깊이 = 2 + retry_count

## 7) 응답 포맷

- 한국어 고정
- 점검 절차 + 이유 (우선순위별)
- 멀티홉 경로 요약
- `top_image_path`: Top-1 근거 페이지 이미지
- `top_manual_sources`: 매뉴얼 출처 (최대 5개, 파일/페이지/점수 포함)
- `top_faq_sources`: FAQ 출처 (최대 5개, 질문/답변/카테고리 포함)
- `confidence`: 신뢰도 점수 (`max(0.25, min(0.95, top1_score))`)
- `graph_paths`: 그래프 탐색 경로 (최대 10개, 디버그용)

## 8) 이중 저장소 구조

### Neo4j (매뉴얼 데이터)
- 계층 구조: Brand → Model → Manual → Section → Page → Chunk
- 벡터 인덱스 + 전문 검색 인덱스
- 시맨틱 엔티티 그래프 (Entity, Symptom, Action, DTC)
- 페이지 이미지 참조 (PNG 렌더링)

### ChromaDB (FAQ 데이터)
> 파일: `src/Chevolet_GraphRAG/retrieval/chroma_faq.py`

- 평면 구조: question, answer, category
- 벡터 유사도 검색
- 지속 저장: `artifacts/chroma_faq/`
- 소스: `data/FAQ/chevrolet_faq_target_data.json` (500+ FAQ 항목)

라우팅:
- 가드레일에서 `prefer_faq` 소프트 힌트를 제공하지만 하드 라우팅 아님
- 항상 매뉴얼 + FAQ 모두 검색 → 점수 기반 최종 선택

## 9) 관측성 (Observability)

> 파일: `src/Chevolet_GraphRAG/observability/langsmith_client.py`

- LangSmith 기반 트레이싱
- 워크플로우 전체를 하나의 trace로 감싸서 실행
- trace 시작: `session_id`, `query` 기록
- trace 종료: `confidence`, `retry_count`, `allow` 이벤트 기록
- 환경변수: `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING`

## 10) 주요 외부 의존성

| 컴포넌트 | 라이브러리 |
|----------|-----------|
| LLM | `langchain-openai` (GPT-4.1-mini) 또는 `langchain-ollama` |
| 임베딩 | `langchain-huggingface` (BAAI/bge-m3, 1024차원) |
| 리랭커 | `langchain-cohere` (rerank-v3.5) 또는 HuggingFace CrossEncoder |
| 그래프 DB | `neo4j` + `langchain-neo4j` |
| FAQ 벡터 DB | `chromadb` |
| 워크플로우 | `langgraph` |
| PDF 파싱 | `pymupdf` (fitz) + `langchain-docling` (OCR 폴백) |
| 모델 매칭 | `rapidfuzz` |
| API 서버 | `fastapi` + `uvicorn` |
| 트레이싱 | `langsmith` |
