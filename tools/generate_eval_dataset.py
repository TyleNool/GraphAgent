#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from Chevolet_GraphRAG.ingest import discover_manual_files
from Chevolet_GraphRAG.retrieval.guardrails import (
    MODEL_FAMILY_ALIASES,
    MODEL_KO_EN_ALIASES,
    MODEL_VARIANT_PREFIXES,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
FAQ_PATH = DATA_ROOT / "FAQ" / "chevrolet_faq_target_data.json"
OUTPUT_PATH = PROJECT_ROOT / "Comprehensive_GraphRAG_Evaluation_Dataset_300.json"

SEED = 20260318


TOPIC_SPECS: dict[str, list[dict[str, Any]]] = {
    "overview": [
        {
            "question": "기본 특징과 차량 개요를 알려주세요",
            "keywords": ["개요", "차량", "소개", "특징"],
            "facts": ["개요", "차량", "특징"],
            "prohibited": ["임의 개조", "무제한 성능 향상"],
        }
    ],
    "cluster_controls": [
        {
            "question": "경고등과 계기판 표시 의미를 확인하고 싶습니다",
            "keywords": ["경고등", "계기판", "표시등", "체크엔진", "MIL"],
            "facts": ["경고등", "계기판", "표시등"],
            "prohibited": ["경고등을 무시하고 계속 운행", "계기판 점검 불필요"],
        }
    ],
    "driving_operation": [
        {
            "question": "시동과 기본 주행 조작 순서를 알려주세요",
            "keywords": ["시동", "주행", "브레이크", "변속", "운전"],
            "facts": ["시동", "주행", "브레이크"],
            "prohibited": ["브레이크 없이 시동", "임의 변속 조작"],
        }
    ],
    "infotainment": [
        {
            "question": "인포테인먼트와 블루투스 사용법을 알려주세요",
            "keywords": ["인포테인먼트", "블루투스", "오디오", "라디오", "업데이트"],
            "facts": ["인포테인먼트", "블루투스", "오디오"],
            "prohibited": ["주행 중 모든 기능 조작 가능", "업데이트 불필요"],
        }
    ],
    "access_openings": [
        {
            "question": "스마트키와 도어, 유리창 사용법을 알려주세요",
            "keywords": ["키", "도어", "유리창", "창문", "트렁크"],
            "facts": ["키", "도어", "유리창"],
            "prohibited": ["문이 열린 상태로 주행", "스마트키 점검 불필요"],
        }
    ],
    "lighting": [
        {
            "question": "조명과 등화장치 조작 방법을 알려주세요",
            "keywords": ["조명", "등화", "전조등", "헤드램프", "램프"],
            "facts": ["조명", "전조등", "램프"],
            "prohibited": ["등화장치 고장 무시", "임의 전구 규격 사용"],
        }
    ],
    "hvac": [
        {
            "question": "에어컨과 히터, 성에 제거 사용법을 알려주세요",
            "keywords": ["에어컨", "히터", "성에", "송풍", "재순환"],
            "facts": ["에어컨", "히터", "성에"],
            "prohibited": ["과열 상태에서 에어컨 계속 사용", "성에 제거 불필요"],
        }
    ],
    "seat_safety": [
        {
            "question": "시트와 안전장치 사용법을 알려주세요",
            "keywords": ["시트", "안전벨트", "에어백", "안전", "유아용"],
            "facts": ["시트", "안전벨트", "에어백"],
            "prohibited": ["안전벨트 없이 주행", "에어백 점검 불필요"],
        }
    ],
    "service_maintenance": [
        {
            "question": "정기 점검 항목과 기본 정비 절차를 알려주세요",
            "keywords": ["정기", "점검", "정비", "오일", "타이어", "브레이크"],
            "facts": ["점검", "정비", "오일"],
            "prohibited": ["오일 교환 불필요", "타이어 점검 생략"],
        }
    ],
    "vehicle_care": [
        {
            "question": "차량 관리와 세차, 보호 요령을 알려주세요",
            "keywords": ["차량", "관리", "세차", "부식", "보호"],
            "facts": ["차량", "관리", "세차"],
            "prohibited": ["염분 제거 불필요", "부식 점검 불필요"],
        }
    ],
    "storage": [
        {
            "question": "장기 보관 전에 확인할 사항을 알려주세요",
            "keywords": ["보관", "장기", "배터리", "연료", "타이어"],
            "facts": ["보관", "배터리", "타이어"],
            "prohibited": ["장기 보관 전 점검 불필요", "배터리 분리 불필요"],
        }
    ],
    "warranty": [
        {
            "question": "보증 범위와 보증서 확인 방법을 알려주세요",
            "keywords": ["보증", "보증서", "서비스", "기간", "수리"],
            "facts": ["보증", "보증서", "기간"],
            "prohibited": ["모든 소모품 무상", "보증 조건 없음"],
        }
    ],
    "specs": [
        {
            "question": "제원과 규격 정보를 알려주세요",
            "keywords": ["제원", "규격", "용량", "공기압", "연료"],
            "facts": ["제원", "규격", "용량"],
            "prohibited": ["임의 규격 사용 가능", "공기압 확인 불필요"],
        }
    ],
    "emergency_action": [
        {
            "question": "긴급 상황 시 조치 방법을 알려주세요",
            "keywords": ["긴급", "비상", "점프", "견인", "응급"],
            "facts": ["긴급", "점프", "견인"],
            "prohibited": ["견인 고리 없이 견인", "배터리 점프 불필요"],
        }
    ],
    "general": [
        {
            "question": "차량 사용 관련 기본 안내를 알려주세요",
            "keywords": ["차량", "사용", "안내"],
            "facts": ["차량", "안내"],
            "prohibited": ["임의 조작 권장"],
        }
    ],
}


FAQ_MIX_CATEGORY_TO_MANUAL_TYPES: dict[str, list[str]] = {
    "MyLink": ["infotainment"],
    "내비 업데이트": ["infotainment"],
    "Android Auto/\nApple CarPlay": ["infotainment"],
    "차량 관리": ["vehicle_care", "service_maintenance", "storage"],
    "EV 리콜": ["emergency_action", "service_maintenance"],
    "구매 관련": ["warranty", "service_maintenance"],
    "통합계정 및 홈페이지 이용": ["infotainment", "overview"],
}


LOW_CONF_QUERY_SUFFIXES = [
    "관련해서 한 번에 다 정리해 주세요",
    "중요한 것들만 폭넓게 알려주세요",
    "어디부터 봐야 할지 모르겠는데 전체적으로 설명해 주세요",
]


QUESTION_UNIQUIFIER_SUFFIXES: dict[str, list[str]] = {
    "Guardrail_Positive": [
        "쉐보레 차량 기준으로 설명해주세요",
        "기본 사용 상황부터 알려주세요",
        "운전자 입장에서 핵심만 알려주세요",
        "주의사항까지 포함해 주세요",
    ],
    "Model_Disambiguation": [
        "차종 구분이 필요하면 함께 짚어주세요",
        "동일 계열 모델과 혼동되지 않게 설명해주세요",
        "어떤 모델 후보인지 먼저 정리해 주세요",
        "모델 식별 관점에서 답해주세요",
    ],
    "Manual_Retrieval_PageAligned": [
        "매뉴얼 기준 핵심 절차만 알려주세요",
        "조작 순서 위주로 정리해주세요",
        "중요 경고까지 함께 알려주세요",
        "운전자 안내 문구 중심으로 설명해주세요",
    ],
    "Manual_Retrieval_MultiHop": [
        "두 정보의 연결 관계가 드러나게 답해주세요",
        "원인과 조치를 함께 이어서 설명해주세요",
        "관련 절차를 순서대로 묶어주세요",
    ],
    "FAQ_Retrieval_Pure": [
        "공식 정책 기준으로 답해주세요",
        "적용 대상과 조건을 같이 알려주세요",
        "고객 안내 문구처럼 간단히 정리해주세요",
        "예외 사항이 있으면 함께 적어주세요",
    ],
    "FAQ_Manual_Mixed": [
        "FAQ와 차량 사용 정보를 구분해서 알려주세요",
        "정책 정보와 차량 안내를 나눠서 설명해주세요",
        "공식 FAQ와 매뉴얼 내용을 함께 정리해주세요",
    ],
    "Source_Selection_Hard": [
        "가장 적절한 출처를 우선해서 답해주세요",
        "정책성 정보인지 차량 사용 정보인지 먼저 판단해주세요",
        "우선 참고해야 할 자료 기준으로 설명해주세요",
    ],
    "Answer_Grounding": [
        "근거 표현 위주로 답해주세요",
        "확인 가능한 사실만 요약해주세요",
        "출처에 없는 추정은 빼고 설명해주세요",
        "핵심 근거만 간단히 정리해주세요",
    ],
    "Confidence_Calibration": [
        "확실한 범위만 답해주세요",
        "불확실한 부분은 단정하지 말아주세요",
        "근거가 명확한 내용만 알려주세요",
        "모르는 부분은 보수적으로 답해주세요",
    ],
}


OTHER_BRAND_QUERIES = [
    "현대 아반떼 타이어 공기압 확인 방법 알려주세요",
    "기아 K5 엔진오일 교환 주기 알려주세요",
    "BMW 3시리즈 경고등 의미를 알고 싶어요",
    "벤츠 E클래스 시동 거는 순서를 알려주세요",
    "아우디 A6 에어컨 필터 교체 방법 알려주세요",
    "렉서스 ES 브레이크 점검 순서 알려주세요",
    "닛산 알티마 블루투스 연결 방법 알려주세요",
    "테슬라 모델3 충전 관련 안내 알려주세요",
    "토요타 캠리 보증 기간 알려주세요",
    "폭스바겐 티구안 조명 조작 방법 알려주세요",
]


NON_AUTO_QUERIES = [
    "주식 포트폴리오 추천해 주세요",
    "코인 투자 전략 알려주세요",
    "부동산 계약서 작성 방법 알려주세요",
    "다이어트 식단 짜 주세요",
    "면접 자기소개서 첨삭해 주세요",
    "연애 상담 부탁합니다",
    "감기 처방약 추천해 주세요",
    "요리 레시피 알려주세요",
    "이력서 문장 교정해 주세요",
    "부동산 세금 계산법 알려주세요",
]


@dataclass(slots=True)
class ManualRecord:
    model: str
    manual_type: str
    file_path: Path
    source_file: str
    family: str


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_token(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", (text or "").lower())


def _strip_variant_prefix(token: str) -> str:
    for prefix in MODEL_VARIANT_PREFIXES:
        if token.startswith(prefix):
            return token[len(prefix):]
    return token


def infer_model_family(value: str | None) -> str:
    token = _normalize_token(value or "")
    if not token:
        return ""
    if token in MODEL_FAMILY_ALIASES:
        return MODEL_FAMILY_ALIASES[token]
    for alias, target in MODEL_KO_EN_ALIASES.items():
        if _normalize_token(alias) == token:
            if _normalize_token(target) == token:
                break
            return infer_model_family(target)
    stripped = _strip_variant_prefix(token)
    if stripped in MODEL_FAMILY_ALIASES:
        return MODEL_FAMILY_ALIASES[stripped]
    return stripped


def display_model_name(model: str) -> str:
    replacements = {
        "ALL_NEW_": "올뉴 ",
        "THE_NEW_": "더뉴 ",
        "The_New_": "더뉴 ",
        "REAL_NEW_": "리얼뉴 ",
        "All_New_": "올뉴 ",
        "Spark_EV": "스파크 EV",
        "Spark": "스파크",
        "BOLT_EV": "볼트 EV",
        "BOLT_EUV": "볼트 EUV",
        "TAHOE": "타호",
        "TRAX_CROSSOVER": "트랙스 크로스오버",
        "VOLT": "Volt",
    }
    if model in replacements:
        return replacements[model]
    for prefix, repl in replacements.items():
        if model.startswith(prefix):
            return model.replace(prefix, repl).replace("_", " ").strip()
    return model.replace("_", " ").strip()


def clean_question_prefix(text: str) -> str:
    text = _normalize_text(text)
    return re.sub(r"^\[[^\]]+\]\s*", "", text).strip()


def answer_snippets(answer: str, limit: int = 2) -> list[str]:
    parts = [
        _normalize_text(part)
        for part in re.split(r"[.!?\n]+", answer)
        if _normalize_text(part)
    ]
    snippets: list[str] = []
    for part in parts:
        if len(part) < 10:
            continue
        snippets.append(part[:120])
        if len(snippets) >= limit:
            break
    return snippets or ([_normalize_text(answer)[:120]] if answer else [])


def ensure_unique_questions(dataset: list[dict[str, Any]]) -> None:
    used: set[str] = set()
    category_counters: dict[str, int] = defaultdict(int)
    for item in dataset:
        base_question = _normalize_text(str(item.get("question", "")))
        if base_question not in used:
            item["question"] = base_question
            used.add(base_question)
            continue

        category = str(item.get("category", ""))
        suffixes = QUESTION_UNIQUIFIER_SUFFIXES.get(category, ["세부 조건까지 함께 알려주세요"])
        category_counters[category] += 1

        chosen = None
        for offset in range(len(suffixes)):
            suffix = suffixes[(category_counters[category] + offset - 1) % len(suffixes)]
            candidate = _normalize_text(f"{base_question} {suffix}")
            if candidate not in used:
                chosen = candidate
                break

        if chosen is None:
            serial = 2
            while True:
                candidate = _normalize_text(f"{base_question} [{category} {serial}]")
                if candidate not in used:
                    chosen = candidate
                    break
                serial += 1

        item["question"] = chosen
        used.add(chosen)


class DatasetBuilder:
    def __init__(self) -> None:
        catalog = discover_manual_files(DATA_ROOT)
        self.manuals = [
            ManualRecord(
                model=m.model,
                manual_type=m.manual_type,
                file_path=m.file_path,
                source_file=m.file_path.relative_to(PROJECT_ROOT).as_posix(),
                family=infer_model_family(m.model),
            )
            for m in catalog.manuals
            if "Zone.Identifier" not in m.file_path.as_posix()
        ]
        self.manuals.sort(key=lambda x: (x.model, x.manual_type, x.source_file))
        self.page_cache: dict[Path, list[str]] = {}
        self.faq_entries = json.loads(FAQ_PATH.read_text(encoding="utf-8"))
        self.family_to_models: dict[str, list[str]] = defaultdict(list)
        self.model_to_manuals: dict[str, list[ManualRecord]] = defaultdict(list)
        self.model_manual_type_to_manuals: dict[tuple[str, str], list[ManualRecord]] = defaultdict(list)
        for manual in self.manuals:
            if manual.model not in self.family_to_models[manual.family]:
                self.family_to_models[manual.family].append(manual.model)
            self.model_to_manuals[manual.model].append(manual)
            self.model_manual_type_to_manuals[(manual.model, manual.manual_type)].append(manual)
        for family in self.family_to_models:
            self.family_to_models[family].sort()

    def load_pages(self, path: Path) -> list[str]:
        cached = self.page_cache.get(path)
        if cached is not None:
            return cached
        with fitz.open(path) as doc:
            pages = [_normalize_text(page.get_text("text")) for page in doc]
        self.page_cache[path] = pages
        return pages

    def best_page(self, manual: ManualRecord, keywords: list[str]) -> tuple[int, list[str], str]:
        pages = self.load_pages(manual.file_path)
        best_index = 0
        best_score = -1
        best_terms: list[str] = []
        for idx, text in enumerate(pages):
            lowered = text.lower()
            present = [kw for kw in keywords if kw.lower() in lowered]
            score = sum(lowered.count(kw.lower()) for kw in keywords)
            if score > best_score:
                best_score = score
                best_index = idx
                best_terms = present
        excerpt = pages[best_index][:240] if pages else ""
        return best_index + 1, best_terms[:4], excerpt

    def manual_topic(self, manual: ManualRecord) -> dict[str, Any]:
        return TOPIC_SPECS.get(manual.manual_type, TOPIC_SPECS["general"])[0]

    def base_manual_item(
        self,
        item_id: str,
        category: str,
        manual: ManualRecord,
        question: str,
        confidence_bucket: str = "high",
        source_family: str = "manual",
        expected_model_candidates: list[str] | None = None,
        extra_manual_types: list[str] | None = None,
        extra_source_files: list[str] | None = None,
        expected_faq_categories: list[str] | None = None,
        should_not_return_manual: bool = False,
        expected_entities: list[str] | None = None,
        expected_relations: list[str] | None = None,
        path_complete: bool = False,
        expected_facts: list[str] | None = None,
        prohibited_facts: list[str] | None = None,
        reference_answer: str = "",
        expected_groundedness_pass: bool = True,
    ) -> dict[str, Any]:
        spec = self.manual_topic(manual)
        page_no, found_terms, excerpt = self.best_page(manual, spec["keywords"])
        facts = list(dict.fromkeys((expected_facts or []) + found_terms + spec["facts"]))[:4]
        entities = list(dict.fromkeys((expected_entities or []) + facts[:3]))[:4]
        model_candidates = expected_model_candidates or self.family_to_models.get(manual.family, [manual.model])
        manual_types = list(dict.fromkeys((extra_manual_types or []) + [manual.manual_type]))
        source_files = list(dict.fromkeys((extra_source_files or []) + [manual.source_file]))
        return {
            "id": item_id,
            "category": category,
            "difficulty": "advanced" if path_complete or len(manual_types) > 1 else "basic",
            "question": question,
            "ground_truth": {
                "guardrail": {
                    "expected_allow": True,
                    "expected_reject_reason": None,
                    "expected_source_family": source_family,
                    "expected_model_family": model_candidates[0] if model_candidates else manual.model,
                    "expected_model_candidates": model_candidates,
                },
                "retrieval": {
                    "expected_source_files": source_files,
                    "expected_pages": [page_no],
                    "expected_manual_types": manual_types,
                    "expected_faq_categories": expected_faq_categories or [],
                    "expected_image_page": page_no,
                    "should_not_return_manual": should_not_return_manual,
                },
                "answer": {
                    "expected_facts": facts,
                    "prohibited_facts": prohibited_facts or spec["prohibited"],
                    "reference_answer": reference_answer,
                },
                "quality": {
                    "expected_groundedness_pass": expected_groundedness_pass,
                    "expected_confidence_bucket": confidence_bucket,
                },
                "graphrag": {
                    "expected_entities": entities,
                    "expected_relations": expected_relations or [],
                    "expected_path_completeness": path_complete,
                },
            },
            "eval_stage_focus": self.eval_focus_for(
                source_family=source_family,
                manual_types=manual_types,
                has_faq=bool(expected_faq_categories),
                path_complete=path_complete,
                has_facts=bool(facts),
                confidence_bucket=confidence_bucket,
            ),
            "notes": excerpt[:180],
        }

    def faq_item(
        self,
        item_id: str,
        category: str,
        faq_entry: dict[str, Any],
        question: str | None = None,
        confidence_bucket: str = "high",
        source_family: str = "faq",
        should_not_return_manual: bool = True,
        expected_model_candidates: list[str] | None = None,
        expected_model_family: str | None = None,
        expected_manual_types: list[str] | None = None,
        path_complete: bool = False,
    ) -> dict[str, Any]:
        answer = _normalize_text(faq_entry.get("answer", ""))
        facts = answer_snippets(answer)
        return {
            "id": item_id,
            "category": category,
            "difficulty": "basic" if source_family == "faq" else "intermediate",
            "question": question or clean_question_prefix(str(faq_entry.get("question", ""))),
            "ground_truth": {
                "guardrail": {
                    "expected_allow": True,
                    "expected_reject_reason": None,
                    "expected_source_family": source_family,
                    "expected_model_family": expected_model_family,
                    "expected_model_candidates": expected_model_candidates or [],
                },
                "retrieval": {
                    "expected_source_files": [FAQ_PATH.relative_to(PROJECT_ROOT).as_posix()],
                    "expected_pages": [],
                    "expected_manual_types": expected_manual_types or [],
                    "expected_faq_categories": [str(faq_entry.get("category", ""))],
                    "expected_image_page": None,
                    "should_not_return_manual": should_not_return_manual,
                },
                "answer": {
                    "expected_facts": facts,
                    "prohibited_facts": [],
                    "reference_answer": answer,
                },
                "quality": {
                    "expected_groundedness_pass": True,
                    "expected_confidence_bucket": confidence_bucket,
                },
                "graphrag": {
                    "expected_entities": [],
                    "expected_relations": [],
                    "expected_path_completeness": path_complete,
                },
            },
            "eval_stage_focus": self.eval_focus_for(
                source_family=source_family,
                manual_types=expected_manual_types or [],
                has_faq=True,
                path_complete=path_complete,
                has_facts=bool(facts),
                confidence_bucket=confidence_bucket,
            ),
            "notes": answer[:180],
        }

    def eval_focus_for(
        self,
        source_family: str,
        manual_types: list[str],
        has_faq: bool,
        path_complete: bool,
        has_facts: bool,
        confidence_bucket: str,
    ) -> list[str]:
        focus = ["1_guardrail"]
        if source_family in {"manual", "mixed", "faq"}:
            focus.append("2_retrieval")
        if source_family in {"faq", "mixed"} and has_faq:
            focus.append("10_source_selection")
        if path_complete:
            focus.append("9_graphrag")
        if has_facts:
            focus.extend(["3_answer", "5_grounding"])
        if confidence_bucket in {"high", "medium", "low"}:
            focus.append("7_confidence")
        return list(dict.fromkeys(focus))

    def choose_manuals(self, predicate, limit: int) -> list[ManualRecord]:
        picked: list[ManualRecord] = []
        for manual in self.manuals:
            if predicate(manual):
                picked.append(manual)
            if len(picked) >= limit:
                break
        return picked

    def round_robin_faq(self, limit: int) -> list[dict[str, Any]]:
        buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        for entry in self.faq_entries:
            buckets[str(entry.get("category", ""))].append(entry)
        ordered = [buckets[key] for key in sorted(buckets)]
        picked: list[dict[str, Any]] = []
        while ordered and len(picked) < limit:
            next_round: list[deque[dict[str, Any]]] = []
            for queue in ordered:
                if queue:
                    picked.append(queue.popleft())
                    if len(picked) >= limit:
                        break
                if queue:
                    next_round.append(queue)
            ordered = next_round
        return picked

    def mixed_pairs(self) -> list[tuple[dict[str, Any], ManualRecord]]:
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in self.faq_entries:
            by_category[str(entry.get("category", ""))].append(entry)
        pairs: list[tuple[dict[str, Any], ManualRecord]] = []
        for category, manual_types in FAQ_MIX_CATEGORY_TO_MANUAL_TYPES.items():
            faq_entries = by_category.get(category, [])
            if not faq_entries:
                continue
            manual_candidates = [
                manual for manual in self.manuals if manual.manual_type in manual_types
            ]
            for idx, manual in enumerate(manual_candidates):
                pairs.append((faq_entries[idx % len(faq_entries)], manual))
        return pairs


def build_guardrail_negative() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, query in enumerate(OTHER_BRAND_QUERIES + NON_AUTO_QUERIES, start=1):
        reason = "타 브랜드 차량 질의" if idx <= len(OTHER_BRAND_QUERIES) else "비자동차 질의"
        items.append(
            {
                "id": f"GR_NEG_{idx:03d}",
                "category": "Guardrail_Negative",
                "difficulty": "basic",
                "question": query,
                "ground_truth": {
                    "guardrail": {
                        "expected_allow": False,
                        "expected_reject_reason": reason,
                        "expected_source_family": "none",
                        "expected_model_family": None,
                        "expected_model_candidates": [],
                    },
                    "retrieval": {
                        "expected_source_files": [],
                        "expected_pages": [],
                        "expected_manual_types": [],
                        "expected_faq_categories": [],
                        "expected_image_page": None,
                        "should_not_return_manual": False,
                    },
                    "answer": {
                        "expected_facts": [],
                        "prohibited_facts": [],
                        "reference_answer": "",
                    },
                    "quality": {
                        "expected_groundedness_pass": False,
                        "expected_confidence_bucket": "low",
                    },
                    "graphrag": {
                        "expected_entities": [],
                        "expected_relations": [],
                        "expected_path_completeness": False,
                    },
                },
                "eval_stage_focus": ["1_guardrail"],
                "notes": reason,
            }
        )
    return items


def build_guardrail_positive(builder: DatasetBuilder) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    manuals = builder.choose_manuals(lambda m: m.manual_type in {"driving_operation", "service_maintenance", "hvac", "infotainment"}, 12)
    for idx, manual in enumerate(manuals, start=1):
        question = f"{display_model_name(manual.model)} {builder.manual_topic(manual)['question']}"
        items.append(
            builder.base_manual_item(
                item_id=f"GR_POS_{idx:03d}",
                category="Guardrail_Positive",
                manual=manual,
                question=question,
                confidence_bucket="high",
            )
        )
    for offset, faq_entry in enumerate(builder.round_robin_faq(8), start=len(items) + 1):
        items.append(
            builder.faq_item(
                item_id=f"GR_POS_{offset:03d}",
                category="Guardrail_Positive",
                faq_entry=faq_entry,
                confidence_bucket="high",
            )
        )
    return items


def build_model_disambiguation(builder: DatasetBuilder) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    families = [
        (family, models)
        for family, models in sorted(builder.family_to_models.items())
        if len(models) > 1
    ]
    idx = 1
    while len(items) < 30:
        for family, models in families:
            if len(items) >= 30:
                break
            manual = builder.model_to_manuals[models[0]][0]
            aliases = [display_model_name(model) for model in models]
            aliases.append(display_model_name(models[0]).replace("올뉴 ", "").replace("더뉴 ", "").replace("리얼뉴 ", "").strip())
            for alias in list(dict.fromkeys(a for a in aliases if a))[:5]:
                if len(items) >= 30:
                    break
                question = f"{alias} {builder.manual_topic(manual)['question']}"
                items.append(
                    builder.base_manual_item(
                        item_id=f"MODEL_DIS_{idx:03d}",
                        category="Model_Disambiguation",
                        manual=manual,
                        question=question,
                        confidence_bucket="high",
                        expected_model_candidates=models,
                    )
                )
                idx += 1
    return items


def build_manual_page_aligned(builder: DatasetBuilder) -> list[dict[str, Any]]:
    candidates = builder.choose_manuals(
        lambda m: m.manual_type not in {"general"},
        60,
    )
    items: list[dict[str, Any]] = []
    for idx, manual in enumerate(candidates, start=1):
        question = f"{display_model_name(manual.model)} {builder.manual_topic(manual)['question']}"
        items.append(
            builder.base_manual_item(
                item_id=f"MAN_PAGE_{idx:03d}",
                category="Manual_Retrieval_PageAligned",
                manual=manual,
                question=question,
                confidence_bucket="high",
            )
        )
    return items


def build_manual_multihop(builder: DatasetBuilder) -> list[dict[str, Any]]:
    preferred_pairs = [
        ("cluster_controls", "service_maintenance"),
        ("emergency_action", "driving_operation"),
        ("infotainment", "service_maintenance"),
        ("hvac", "service_maintenance"),
        ("vehicle_care", "storage"),
    ]
    items: list[dict[str, Any]] = []
    idx = 1
    for model in sorted(builder.model_to_manuals):
        available_types = {manual.manual_type for manual in builder.model_to_manuals[model]}
        for t1, t2 in preferred_pairs:
            if len(items) >= 30:
                break
            if t1 not in available_types or t2 not in available_types:
                continue
            manual1 = builder.model_manual_type_to_manuals[(model, t1)][0]
            manual2 = builder.model_manual_type_to_manuals[(model, t2)][0]
            spec1 = builder.manual_topic(manual1)
            spec2 = builder.manual_topic(manual2)
            question = (
                f"{display_model_name(model)} "
                f"{spec1['facts'][0]} 관련 의미와 {spec2['facts'][0]} 점검/조치 순서를 함께 알려주세요"
            )
            expected_entities = list(dict.fromkeys(spec1["facts"][:2] + spec2["facts"][:2]))
            items.append(
                builder.base_manual_item(
                    item_id=f"MAN_HOP_{idx:03d}",
                    category="Manual_Retrieval_MultiHop",
                    manual=manual1,
                    question=question,
                    confidence_bucket="medium",
                    extra_manual_types=[manual2.manual_type],
                    extra_source_files=[manual2.source_file],
                    expected_entities=expected_entities,
                    expected_relations=["HAS_SYMPTOM", "RESOLVED_BY"],
                    path_complete=True,
                    expected_facts=expected_entities,
                    prohibited_facts=["근거 없이 단정", "임의 교체 권장"],
                )
            )
            idx += 1
        if len(items) >= 30:
            break
    return items[:30]


def build_faq_pure(builder: DatasetBuilder) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, faq_entry in enumerate(builder.round_robin_faq(40), start=1):
        items.append(
            builder.faq_item(
                item_id=f"FAQ_PURE_{idx:03d}",
                category="FAQ_Retrieval_Pure",
                faq_entry=faq_entry,
                confidence_bucket="high",
            )
        )
    return items


def build_faq_manual_mixed(builder: DatasetBuilder) -> list[dict[str, Any]]:
    pairs = builder.mixed_pairs()[:20]
    items: list[dict[str, Any]] = []
    for idx, (faq_entry, manual) in enumerate(pairs, start=1):
        faq_category = str(faq_entry.get("category", ""))
        question = (
            f"{display_model_name(manual.model)} "
            f"{clean_question_prefix(str(faq_entry.get('question', '')))} "
            f"그리고 {builder.manual_topic(manual)['facts'][0]} 관련 안내도 같이 알려주세요"
        )
        items.append(
            builder.base_manual_item(
                item_id=f"MIX_{idx:03d}",
                category="FAQ_Manual_Mixed",
                manual=manual,
                question=question,
                confidence_bucket="medium",
                source_family="mixed",
                expected_faq_categories=[faq_category],
                extra_source_files=[FAQ_PATH.relative_to(PROJECT_ROOT).as_posix()],
                expected_facts=answer_snippets(_normalize_text(str(faq_entry.get("answer", ""))), limit=1),
                prohibited_facts=[],
                reference_answer=_normalize_text(str(faq_entry.get("answer", ""))),
            )
        )
    return items


def build_source_selection_hard(builder: DatasetBuilder) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    faq_entries = builder.round_robin_faq(10)
    for idx, faq_entry in enumerate(faq_entries, start=1):
        question = f"{clean_question_prefix(str(faq_entry.get('question', '')))} 관련 정책만 알려주세요"
        items.append(
            builder.faq_item(
                item_id=f"SRC_SEL_{idx:03d}",
                category="Source_Selection_Hard",
                faq_entry=faq_entry,
                question=question,
                confidence_bucket="medium",
            )
        )

    manual_candidates = builder.choose_manuals(
        lambda m: m.manual_type in {"warranty", "infotainment", "service_maintenance", "vehicle_care"},
        10,
    )
    base = len(items)
    for offset, manual in enumerate(manual_candidates, start=1):
        question = f"{display_model_name(manual.model)} {builder.manual_topic(manual)['question']}"
        items.append(
            builder.base_manual_item(
                item_id=f"SRC_SEL_{base + offset:03d}",
                category="Source_Selection_Hard",
                manual=manual,
                question=question,
                confidence_bucket="medium",
            )
        )
    return items


def build_answer_grounding(builder: DatasetBuilder) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    manual_candidates = builder.choose_manuals(
        lambda m: m.manual_type in {"service_maintenance", "cluster_controls", "hvac", "specs", "emergency_action"},
        15,
    )
    for idx, manual in enumerate(manual_candidates, start=1):
        spec = builder.manual_topic(manual)
        question = f"{display_model_name(manual.model)} {spec['question']}"
        items.append(
            builder.base_manual_item(
                item_id=f"ANS_GRD_{idx:03d}",
                category="Answer_Grounding",
                manual=manual,
                question=question,
                confidence_bucket="high",
                expected_facts=spec["facts"][:3],
                prohibited_facts=spec["prohibited"],
            )
        )
    faq_entries = builder.round_robin_faq(15)
    base = len(items)
    for offset, faq_entry in enumerate(faq_entries, start=1):
        items.append(
            builder.faq_item(
                item_id=f"ANS_GRD_{base + offset:03d}",
                category="Answer_Grounding",
                faq_entry=faq_entry,
                confidence_bucket="high",
            )
        )
    return items


def build_confidence_calibration(builder: DatasetBuilder) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    high_manuals = builder.choose_manuals(
        lambda m: m.manual_type in {"driving_operation", "hvac", "infotainment", "service_maintenance", "cluster_controls"},
        10,
    )
    for idx, manual in enumerate(high_manuals, start=1):
        question = f"{display_model_name(manual.model)} {builder.manual_topic(manual)['question']}"
        items.append(
            builder.base_manual_item(
                item_id=f"CONF_{idx:03d}",
                category="Confidence_Calibration",
                manual=manual,
                question=question,
                confidence_bucket="high",
            )
        )

    medium_pairs = build_faq_manual_mixed(builder)[:10]
    base = len(items)
    for offset, item in enumerate(medium_pairs, start=1):
        item["id"] = f"CONF_{base + offset:03d}"
        item["category"] = "Confidence_Calibration"
        item["ground_truth"]["quality"]["expected_confidence_bucket"] = "medium"
        item["notes"] = f"mixed confidence case | {item.get('notes', '')}"[:180]
        items.append(item)

    low_manuals = builder.choose_manuals(
        lambda m: m.manual_type in {"general", "overview", "warranty", "vehicle_care", "storage"},
        10,
    )
    base = len(items)
    for offset, manual in enumerate(low_manuals, start=1):
        suffix = LOW_CONF_QUERY_SUFFIXES[(offset - 1) % len(LOW_CONF_QUERY_SUFFIXES)]
        question = f"{display_model_name(manual.model)} {suffix}"
        items.append(
            builder.base_manual_item(
                item_id=f"CONF_{base + offset:03d}",
                category="Confidence_Calibration",
                manual=manual,
                question=question,
                confidence_bucket="low",
                expected_facts=[],
                prohibited_facts=[],
            )
        )
    return items


def build_dataset() -> dict[str, Any]:
    builder = DatasetBuilder()
    sections = [
        build_guardrail_positive(builder),
        build_guardrail_negative(),
        build_model_disambiguation(builder),
        build_manual_page_aligned(builder),
        build_manual_multihop(builder),
        build_faq_pure(builder),
        build_faq_manual_mixed(builder),
        build_source_selection_hard(builder),
        build_answer_grounding(builder),
        build_confidence_calibration(builder),
    ]
    dataset = [item for section in sections for item in section]
    ensure_unique_questions(dataset)
    counts = Counter(item["category"] for item in dataset)
    duplicate_questions = sum(
        1 for count in Counter(item["question"] for item in dataset).values() if count > 1
    )
    assert len(dataset) == 300, f"expected 300 items, got {len(dataset)}"
    assert duplicate_questions == 0, f"expected unique questions, found {duplicate_questions}"
    return {
        "metadata": {
            "title": "GraphRAG Evaluation Dataset",
            "version": "v1-300",
            "seed": SEED,
            "target_count": 300,
            "generated_count": len(dataset),
            "categories": dict(counts),
            "notes": [
                "Built from local manual PDFs and FAQ JSON.",
                "Manual items include expected_pages and expected_image_page.",
                "FAQ items include expected_faq_categories and reference_answer.",
                "All questions are de-duplicated across the 300-item set.",
            ],
        },
        "dataset": dataset,
    }


def main() -> None:
    payload = build_dataset()
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {OUTPUT_PATH}")
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
