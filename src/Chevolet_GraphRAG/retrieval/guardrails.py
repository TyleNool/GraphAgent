from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz, process

from Chevolet_GraphRAG.config import Settings
from Chevolet_GraphRAG.ingest.catalog import DataCatalog, infer_vehicle_category
from Chevolet_GraphRAG.models import GuardrailDecision
from Chevolet_GraphRAG.providers import build_chat_model, invoke_json


OTHER_BRANDS = [
    "현대",
    "기아",
    "제네시스",
    "르노",
    "쌍용",
    "KG",
    "토요타",
    "렉서스",
    "혼다",
    "닛산",
    "BMW",
    "벤츠",
    "아우디",
    "폭스바겐",
    "Tesla",
    "테슬라",
]

NON_AUTOMOTIVE_PATTERNS = [
    re.compile(r"주식|코인|부동산|의학|처방|진단서", re.IGNORECASE),
    re.compile(r"요리|레시피|다이어트|연애|면접|이력서", re.IGNORECASE),
]

CATEGORY_HINTS = {
    "ev": ["전기", "EV", "배터리"],
    "suv": ["SUV", "스포츠유틸리티", "패밀리카"],
    "sedan": ["세단", "승용"],
    "truck": ["트럭", "픽업"],
}

FAQ_INTENT_HINTS = [
    "faq",
    "포인트",
    "오토포인트",
    "선포인트",
    "오토카드",
    "가족 포인트",
    "가족포인트",
    "합산",
    "온라인샵",
    "online shop",
    "온라인 구매",
    "구매혜택",
    "멤버스",
    "회원가입",
    "로그인",
    "비밀번호",
    "계정",
    "홈페이지",
    "mychevrolet",
    "마이쉐보레",
    "보증",
    "보증기간",
    "프로모션",
    "할인",
    "출고",
    "계약",
    "리콜",
    "내비 업데이트",
    "카플레이",
    "android auto",
    "apple carplay",
    "onlineshop",
    "견적 임시저장",
    "견적임시저장",
    "대리인 출고",
    "직접 출고",
    "법인 할부",
    "현금 구매",
    "포인트카드",
    "근저당 설정 해지",
    "계약금 환불",
    "출고 예정일",
    "중도 상환",
    "통합계정",
    "gm 통합계정",
    "이메일 찾기",
    "계정 삭제",
    "회원 삭제",
    "홈페이지 로그인",
    "차량 추가 등록",
    "온스타 서비스",
    "텔레나브",
    "루센",
    "sd카드 구매"
]

MANUAL_TYPE_HINTS = {
    "cluster_controls": ["경고등", "체크엔진", "계기판", "표시등", "mil", "lamp"],
    "service_maintenance": ["정비", "서비스", "오일", "점검", "교체", "배터리", "퓨즈"],
    "hvac": ["에어컨", "히터", "냉방", "난방", "공조"],
    "infotainment": ["블루투스", "내비", "내비게이션", "오디오", "인포테인먼트", "mylink"],
    "lighting": ["전조등", "헤드램프", "실내등", "조명", "램프"],
    "seat_safety": ["에어백", "벨트", "시트", "안전장치"],
    "driving_operation": ["변속", "주행", "크루즈", "핸들", "브레이크", "가속"],
    "vehicle_care": ["세차", "왁스", "차량관리", "부식"],
    "storage": ["보관", "장기보관"],
    "specs": ["제원", "규격", "용량"],
    "access_openings": ["키", "도어", "문", "유리창", "창문", "윈도우", "트렁크"],
    "emergency_action": ["긴급", "견인", "점프", "비상", "응급", "고장"],
}

MODEL_VARIANT_PREFIXES = (
    "allnew",
    "thenew",
    "realnew",
    "올뉴",
    "더뉴",
    "리얼뉴",
)

MODEL_FAMILY_ALIASES: dict[str, str] = {
    "malibu": "말리부",
    "alpheon": "알페온",
    "aveo": "아베오",
    "camaro": "카마로",
    "captiva": "캡티바",
    "colorado": "콜로라도",
    "cruze": "크루즈",
    "equinox": "이쿼녹스",
    "impala": "임팔라",
    "orlando": "올란도",
    "spark": "spark",
    "sparkev": "sparkev",
    "tahoe": "tahoe",
    "trailblazer": "트레일블레이저",
    "traverse": "트래버스",
    "trax": "트랙스",
    "traxcrossover": "traxcrossover",
    "volt": "volt",
    "boltev": "boltev",
    "bolteuv": "bolteuv",
}

# Korean ↔ English model name aliases
# Keys: Korean names users might type, Values: English directory names in data/
MODEL_KO_EN_ALIASES: dict[str, str] = {
    "올뉴말리부" : "ALL_NEW_말리부",
    "올 뉴 말리부" : "ALL_NEW_말리부",
    "올뉴 말리부" : "ALL_NEW_말리부",
    "올 뉴말리부" : "ALL_NEW_말리부",
    "올뉴카마로SS" : "ALL_NEW_카마로_SS",
    "올 뉴카마로SS" : "ALL_NEW_카마로_SS",
    "올뉴 카마로SS" : "ALL_NEW_카마로_SS",
    "올뉴카마로 SS" : "ALL_NEW_카마로_SS",
    "올 뉴 카마로SS" : "ALL_NEW_카마로_SS",
    "올 뉴카마로 SS" : "ALL_NEW_카마로_SS",
    "올뉴 카마로 SS" : "ALL_NEW_카마로_SS",
    "올 뉴 카마로 SS" : "ALL_NEW_카마로_SS",
    "올뉴콜로라도" : "ALL_NEW_콜로라도",
    "올 뉴 콜로라도" : "ALL_NEW_콜로라도",
    "올뉴 콜로라도" : "ALL_NEW_콜로라도",
    "올 뉴콜로라도" : "ALL_NEW_콜로라도",
    "올뉴크루즈" : "All_New_크루즈",
    "올 뉴 크루즈" : "All_New_크루즈",
    "올뉴 크루즈" : "All_New_크루즈",
    "올 뉴크루즈" : "All_New_크루즈",
    "볼트 ev" : "BOLT_EV",
    "볼트ev" : "BOLT_EV",
    "볼트 euv" : "BOLT_EUV",
    "볼트euv" : "BOLT_EUV",
    "리얼뉴콜로라도" : "REAL_NEW_콜로라도",
    "리얼 뉴콜로라도" : "REAL_NEW_콜로라도",
    "리얼뉴 콜로라도" : "REAL_NEW_콜로라도",
    "리얼 뉴 콜로라도" : "REAL_NEW_콜로라도",
    "스파크" : "Spark",
    "스파크ev": "Spark_EV",
    "스파크 ev": "Spark_EV",
    "타호": "TAHOE",
    "더뉴스파크" : "The_New_Spark",
    "더 뉴스파크" : "The_New_Spark",
    "더뉴 스파크" : "The_New_Spark",
    "더 뉴 스파크" : "The_New_Spark",
    "더뉴말리부" : "THE_NEW_말리부",
    "더 뉴말리부" : "THE_NEW_말리부",
    "더뉴 말리부" : "THE_NEW_말리부",
    "더 뉴 말리부" : "THE_NEW_말리부",
    "더뉴아베오" : "The_New_아베오",
    "더 뉴아베오" : "The_New_아베오",
    "더뉴 아베오" : "The_New_아베오",
    "더 뉴 아베오" : "The_New_아베오",
    "더뉴카마로SS" : "THE_NEW_카마로_SS",
    "더 뉴카마로SS" : "THE_NEW_카마로_SS",
    "더뉴 카마로SS" : "THE_NEW_카마로_SS",
    "더뉴카마로 SS" : "THE_NEW_카마로_SS",
    "더 뉴 카마로SS" : "THE_NEW_카마로_SS",
    "더 뉴카마로 SS" : "THE_NEW_카마로_SS",
    "더뉴 카마로 SS" : "THE_NEW_카마로_SS",
    "더 뉴 카마로 SS" : "THE_NEW_카마로_SS",
    "더뉴트랙스" : "THE_NEW_트랙스",
    "더 뉴트랙스" : "THE_NEW_트랙스",
    "더뉴 트랙스" : "THE_NEW_트랙스",
    "더 뉴 트랙스" : "THE_NEW_트랙스",
    "더뉴트레일블레이저" : "THE_NEW_트레일블레이저",
    "더 뉴트레일블레이저" : "THE_NEW_트레일블레이저",
    "더뉴 트레일블레이저" : "THE_NEW_트레일블레이저",
    "더 뉴 트레일블레이저" : "THE_NEW_트레일블레이저",
    "트랙스 크로스오버": "TRAX_CROSSOVER",
    "트랙스크로스오버": "TRAX_CROSSOVER",
    "볼트" : "VOLT",
    "말리부" : "말리부",
    "알페온": "알페온",
    "올란도": "올란도",
    "이쿼녹스": "이쿼녹스",
    "임팔라": "임팔라",
    "카마로": "카마로",
    "캡티바": "캡티바",
    "크루즈": "크루즈",
    "트래버스": "트래버스",
    "트레일블레이저": "트레일블레이저"
}


@dataclass(slots=True)
class GuardrailEngine:
    settings: Settings
    catalog: DataCatalog
    chat_model: Any = field(init=False, repr=False, default=None)
    known_models: list[str] = field(init=False, repr=False, default_factory=list)
    model_family_to_models: dict[str, list[str]] = field(
        init=False, repr=False, default_factory=dict
    )
    model_to_category: dict[str, str] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.chat_model = build_chat_model(self.settings)
        self.known_models = self.catalog.known_models
        family_to_models: dict[str, list[str]] = {}
        for model in self.known_models:
            family = self._infer_model_family_key(model)
            family_to_models.setdefault(family, []).append(model)
        self.model_family_to_models = {
            family: sorted(models)
            for family, models in family_to_models.items()
        }
        self.model_to_category = self.catalog.model_to_category

    def evaluate(self, query: str, model_hint: str | None = None) -> GuardrailDecision:
        text = query.strip()
        hint = (model_hint or "").strip()
        preferred_manual_types = self._infer_manual_types(text)
        prefer_faq = self._infer_faq_intent(text)

        brand_violation = self._detect_other_brand(text)
        if brand_violation:
            return GuardrailDecision(
                allow=False,
                reason=f"타 브랜드 요청 감지: {brand_violation}",
                model_candidates=[],
                preferred_manual_types=preferred_manual_types,
                prefer_faq=False,
            )

        for pattern in NON_AUTOMOTIVE_PATTERNS:
            if pattern.search(text):
                return GuardrailDecision(
                allow=False,
                reason="차량 진단 범위를 벗어난 요청",
                model_candidates=[],
                preferred_manual_types=preferred_manual_types,
                prefer_faq=False,
            )

        # 1) hint(드롭다운)가 있으면 먼저 resolve
        normalized_model = None
        if hint:
            normalized_model = self._resolve_model_name(hint)

        # 2) hint에서 못 찾으면 쿼리 텍스트에서 시도
        if normalized_model is None:
            normalized_model = self._resolve_model_name(text)

        fallback_category = None
        if normalized_model is None:
            fallback_category = self._infer_category(text)

        if prefer_faq and normalized_model is None:
            return GuardrailDecision(
                allow=True,
                reason="쉐보레 브랜드 FAQ 요청",
                normalized_model=None,
                model_candidates=[],
                fallback_category=None,
                preferred_manual_types=preferred_manual_types,
                prefer_faq=True,
            )

        # 3) 드롭다운에서 모델이 확정된 경우 → LLM 판정 생략, 바로 허용
        if hint and normalized_model is not None:
            return GuardrailDecision(
                allow=True,
                reason=f"드롭다운 모델 선택: {normalized_model}",
                normalized_model=normalized_model,
                model_candidates=self.expand_model_candidates(normalized_model),
                preferred_manual_types=preferred_manual_types,
                prefer_faq=prefer_faq,
            )

        # 4) 모델이 없는 경우만 LLM judge 사용
        llm_decision = self._llm_judge(
            query=text,
            normalized_model=normalized_model,
            fallback_category=fallback_category,
        )

        if llm_decision is not None:
            llm_decision.preferred_manual_types = preferred_manual_types
            llm_decision.model_candidates = self.expand_model_candidates(
                llm_decision.normalized_model or normalized_model
            )
            llm_decision.prefer_faq = prefer_faq
            return llm_decision

        if normalized_model is None:
            return GuardrailDecision(
                allow=True,
                reason="데이터 내 동일 차종 카테고리로 대체 검색",
                normalized_model=None,
                model_candidates=[],
                fallback_category=fallback_category or "unknown",
                preferred_manual_types=preferred_manual_types,
                prefer_faq=prefer_faq,
            )

        return GuardrailDecision(
            allow=True,
            reason="쉐보레 모델로 확인",
            normalized_model=normalized_model,
            model_candidates=self.expand_model_candidates(normalized_model),
            preferred_manual_types=preferred_manual_types,
            prefer_faq=prefer_faq,
        )

    def _detect_other_brand(self, text: str) -> str | None:
        lowered = text.lower()
        for brand in OTHER_BRANDS:
            if brand.lower() in lowered:
                return brand
        return None

    def _resolve_model_name(self, text: str) -> str | None:
        if not text:
            return None

        # 1) Direct substring match against known_models
        for model in self.known_models:
            if model.lower() in text.lower():
                return model

        # 2) Korean alias lookup
        lowered = text.lower()
        for ko_name, en_name in MODEL_KO_EN_ALIASES.items():
            if ko_name in lowered:
                # Verify the alias target exists in known_models
                for model in self.known_models:
                    if en_name.lower() in model.lower() or model.lower() in en_name.lower():
                        return model
                # Return the alias directly if not found in known_models
                return en_name

        # 3) Fuzzy match fallback
        candidate = process.extractOne(
            text,
            self.known_models,
            scorer=fuzz.partial_ratio,
        )
        if candidate and candidate[1] >= 78:
            return str(candidate[0])
        return None

    def expand_model_candidates(self, normalized_model: str | None) -> list[str]:
        if not normalized_model:
            return []

        family = self._infer_model_family_key(normalized_model)
        if not family:
            return []

        candidates = self.model_family_to_models.get(family, [])
        if candidates:
            return candidates

        direct_matches = [
            model
            for model in self.known_models
            if self._normalize_model_token(model) == self._normalize_model_token(normalized_model)
        ]
        return sorted(set(direct_matches))

    def _normalize_model_token(self, text: str) -> str:
        return re.sub(r"[^0-9a-z가-힣]+", "", (text or "").lower())

    def _strip_variant_prefix(self, token: str) -> str:
        stripped = token
        for prefix in MODEL_VARIANT_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                break
        return stripped

    def _infer_model_family_key(self, value: str | None) -> str:
        token = self._normalize_model_token(value or "")
        if not token:
            return ""

        if token in MODEL_FAMILY_ALIASES:
            return MODEL_FAMILY_ALIASES[token]

        for alias, target in MODEL_KO_EN_ALIASES.items():
            if self._normalize_model_token(alias) == token:
                if self._normalize_model_token(target) == token:
                    break
                return self._infer_model_family_key(target)

        stripped = self._strip_variant_prefix(token)
        if stripped in MODEL_FAMILY_ALIASES:
            return MODEL_FAMILY_ALIASES[stripped]
        return stripped

    def _infer_category(self, text: str) -> str:
        upper = text.upper()
        for category, hints in CATEGORY_HINTS.items():
            if any(h.upper() in upper for h in hints):
                return category

        # Try Korean alias → resolve to English model → get category
        lowered = text.lower()
        for ko_name, en_name in MODEL_KO_EN_ALIASES.items():
            if ko_name in lowered:
                return infer_vehicle_category(en_name)

        model = self._resolve_model_name(text)
        if model:
            return self.model_to_category.get(model, "unknown")

        return infer_vehicle_category(text)

    def _infer_manual_types(self, text: str) -> list[str]:
        lowered = text.lower()
        hits: list[str] = []
        for manual_type, hints in MANUAL_TYPE_HINTS.items():
            if any(h.lower() in lowered for h in hints):
                hits.append(manual_type)
        return hits

    def _infer_faq_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(hint.lower() in lowered for hint in FAQ_INTENT_HINTS)

    def _llm_judge(
        self,
        query: str,
        normalized_model: str | None,
        fallback_category: str | None,
    ) -> GuardrailDecision | None:
        if self.chat_model is None:
            return None

        prompt = f"""
        You are a Chevrolet vehicle support request verifier.
        Rules:
        1) allow=false for requests from other brands or non-vehicle.
        2) Chevrolet brand FAQ requests about membership, points, purchase, website/account, warranty, recall, infotainment support are allowed even without a specific model.
        3) Even if the Chevrolet model is not in the data, fallback to the same vehicle category is possible.
        4) However, since the data was loaded in English during the data loading process, the case written in Korean for the model must be matched with the stored data of the model in English.
        5) Only JSON outputs.

        Input :
        - query: {query}
        - normalized_model: {normalized_model}
        - fallback_category: {fallback_category}

        Output Schema:
        {{
        "allow": bool,
        "reason": "...",
        "normalized_model": "... or null",
        "fallback_category": "... or null",
        "requested_action": "answer|reject"
        }}
        """.strip()

        fallback = {
            "allow": True,
            "reason": "rule-based fallback",
            "normalized_model": normalized_model,
            "fallback_category": fallback_category,
            "requested_action": "answer",
        }
        result = invoke_json(self.chat_model, prompt, fallback=fallback)

        try:
            return GuardrailDecision(
                allow=bool(result.get("allow", True)),
                reason=str(result.get("reason", "검증 완료")),
                normalized_model=result.get("normalized_model"),
                fallback_category=result.get("fallback_category"),
                requested_action=str(result.get("requested_action", "answer")),
            )
        except Exception:
            return None
