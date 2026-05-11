"""HyDE — Hypothetical Document Embeddings.

Fluxo:
    1. Recebe query coloquial do usuário.
    2. Gera um documento hipotético em linguagem clínica técnica.
    3. O embedding desse documento hipotético é usado na busca vetorial.

Fallback offline obrigatório — funciona sem OpenAI API.
"""

from __future__ import annotations

import os
import re


# ── Mapeamento semântico: coloquial → terminologia médica ─────────────────────
TERM_MAP: dict[str, str] = {
    # Cefaleia
    "dor de cabeça latejante": "cefaleia pulsátil",
    "dor de cabeça que lateja": "cefaleia pulsátil",
    "dor de cabeça pulsando": "cefaleia pulsátil",
    "dor de cabeça": "cefaleia",
    "cabeça doendo": "cefaleia",
    "latejante": "pulsátil",
    # Fotossensibilidade
    "luz incomodando": "fotofobia",
    "luz na cabeça": "fotofobia",
    "sensível à luz": "fotofobia",
    "incomodo com luz": "fotofobia",
    # Neurológico
    "tontura": "vertigem posicional paroxística benigna",
    "enjoo": "náusea e êmese",
    "tremedeira nas mãos": "tremor de repouso",
    "tremor nas mãos": "tremor de repouso",
    "mão tremendo": "tremor de repouso",
    "esquecendo as coisas": "comprometimento cognitivo / amnésia anterógrada",
    "formigamento nas mãos": "parestesia em extremidades superiores",
    "formigamento nos pés": "parestesia em extremidades inferiores / neuropatia periférica",
    "formigamento": "parestesias / neuropatia periférica",
    "dormência": "hipoestesia",
    "convulsão": "crise epiléptica tônico-clônica generalizada",
    "desmaio": "síncope",
    "fraqueza nas pernas": "paraparesia / déficit motor de membros inferiores",
    "fraqueza muscular": "paresia / miopatia",
    "rigidez": "rigidez muscular (sinal de roda denteada)",
    # Cardiovascular
    "pressão alta": "hipertensão arterial sistêmica",
    "pressão elevada": "hipertensão arterial sistêmica",
    "dor no peito": "dor precordial / angina pectoris",
    "aperto no peito": "dor precordial constritiva / angina",
    "coração acelerado": "taquicardia / palpitações",
    "coração batendo rápido": "taquicardia sinusal / palpitações",
    "inchaço nas pernas": "edema de membros inferiores",
    "inchaço": "edema / anasarca",
    "falta de ar": "dispneia",
    "falta de fôlego": "dispneia aos esforços",
    "cansaço fácil": "fadiga / intolerância ao exercício",
    # Metabólico/endócrino
    "açúcar alto": "hiperglicemia / diabetes mellitus tipo 2",
    "sede excessiva": "polidipsia",
    "urinando muito": "poliúria",
    "ganhando peso": "ganho ponderal",
    "emagrecimento": "perda ponderal involuntária / caquexia",
    "cansado demais": "fadiga crônica / astenia",
    # Respiratório
    "chiado no peito": "sibilância / broncoespasmo",
    "tosse com catarro": "tosse produtiva com expectoração",
    "tosse seca": "tosse seca / irritativa",
    # Gastrointestinal
    "queimação no estômago": "pirose / azia / doença do refluxo gastroesofágico",
    "azia": "pirose / DRGE",
    "náusea": "náusea e vômito",
    "enjoo": "náusea",
    # Sistêmico
    "febre": "hipertermia / síndrome febril",
    "calafrios": "calafrios / rigores",
    "não consegue dormir": "insônia / distúrbio do sono",
    "dor nas juntas": "artralgia / artrite",
    "dor muscular": "mialgia difusa / fibromialgia",
}

# ── Templates clínicos por categoria ─────────────────────────────────────────
TEMPLATES: dict[str, str] = {
    "cefaleia": (
        "Paciente apresenta {terms}. À anamnese, relata crises recorrentes de cefaleia "
        "de moderada a forte intensidade, com caráter {quality}, localização unilateral, "
        "duração de 4 a 72 horas. Associa-se fotofobia, fonofobia, náusea e, em alguns "
        "casos, aura visual (escotoma cintilante). Hipótese diagnóstica: enxaqueca "
        "(migrânea) conforme critérios ICHD-3. Propedêutica: exame neurológico, "
        "ressonância magnética do crânio se atípica. Tratamento: triptanos, AINEs, "
        "profilaxia com topiramato ou propranolol."
    ),
    "cardiovascular": (
        "Paciente apresenta {terms}. Exame físico: pressão arterial elevada, ausculta "
        "cardíaca com B3, presença de crepitações pulmonares bibasais e edema de membros "
        "inferiores. Hipótese diagnóstica: insuficiência cardíaca e/ou hipertensão "
        "arterial sistêmica com lesão de órgão-alvo. Propedêutica: ECG, ecocardiograma, "
        "radiografia de tórax, BNP. Tratamento: IECA/BRA, betabloqueadores, diuréticos."
    ),
    "neurologico": (
        "Paciente com história de {terms}. Exame neurológico evidencia déficits motores "
        "e/ou sensitivos. Hipóteses diagnósticas: neuropatia periférica, doença de "
        "Parkinson, epilepsia ou demência conforme apresentação clínica. "
        "Propedêutica: eletroneuromiografia, ressonância do crânio, EEG."
    ),
    "metabolico": (
        "Paciente apresenta {terms}. Glicemia de jejum e HbA1c solicitadas. "
        "Hipótese diagnóstica: diabetes mellitus tipo 2 com possíveis complicações "
        "microvasculares (retinopatia, nefropatia, neuropatia). Propedêutica: "
        "glicemia, HbA1c, microalbuminúria, creatinina, fundo de olho. "
        "Tratamento: metformina, inibidores SGLT-2, mudança de estilo de vida."
    ),
    "respiratorio": (
        "Paciente com {terms}. Ausculta pulmonar com sibilos difusos e/ou crepitações. "
        "Hipóteses: asma brônquica, DPOC, pneumonia. Espirometria indicada. "
        "Propedêutica: radiografia de tórax, espirometria, gasometria arterial. "
        "Tratamento: broncodilatadores, corticosteroide inalatório, antibióticos se infecção."
    ),
    "default": (
        "Paciente apresenta {terms}. Avaliação clínica completa indica necessidade de "
        "investigação diagnóstica com exames laboratoriais e de imagem. "
        "Hipóteses diagnósticas a serem investigadas conforme apresentação clínica "
        "e histórico médico prévio. Conduta: propedêutica direcionada ao sistema "
        "acometido, suporte sintomático e acompanhamento especializado."
    ),
}


def _map_terms(query: str) -> list[str]:
    """Substitui expressões coloquiais por terminologia médica."""
    query_lower = query.lower()
    found: list[str] = []

    # Ordena por comprimento (maior primeiro) para evitar matches parciais
    for colloquial, medical in sorted(TERM_MAP.items(), key=lambda x: -len(x[0])):
        if colloquial in query_lower:
            if medical not in found:
                found.append(medical)

    if not found:
        # Fallback: retorna a query original em formato clínico
        found = [f"queixas de: {query.lower()}"]

    return found


def _detect_category(mapped_terms: list[str]) -> str:
    """Detecta a categoria clínica predominante a partir dos termos mapeados."""
    text = " ".join(mapped_terms).lower()

    if any(k in text for k in ["cefaleia", "fotofobia", "enxaqueca", "fonofobia", "escotoma"]):
        return "cefaleia"
    # Respiratório verifica antes de cardiovascular pois dispneia aparece nos dois
    if any(k in text for k in ["broncoespasmo", "sibilância", "expectoração"]):
        return "respiratorio"
    if any(k in text for k in ["hipertensão", "taquicardia", "edema", "precordial", "angina"]):
        return "cardiovascular"
    if any(k in text for k in ["tremor", "parestesia", "epiléptica", "sincope", "paresia", "amnésia", "neuropatia", "vertigem"]):
        return "neurologico"
    if any(k in text for k in ["hiperglicemia", "diabetes", "poliúria", "polidipsia"]):
        return "metabolico"
    if any(k in text for k in ["dispneia", "tosse"]):
        return "respiratorio"

    return "default"


def generate_hypothetical_document_offline(query: str) -> str:
    """Gera documento hipotético clínico sem API externa.

    Realiza mapeamento semântico de linguagem coloquial para terminologia médica
    e constrói um caso clínico estruturado usando templates por categoria.
    """
    mapped = _map_terms(query)
    category = _detect_category(mapped)
    terms_str = ", ".join(mapped)

    template = TEMPLATES[category]
    quality = "pulsátil" if "pulsátil" in terms_str else "opressiva"
    document = template.format(terms=terms_str, quality=quality)

    return document


def generate_hypothetical_document_openai(query: str, api_key: str) -> str:
    """Gera documento hipotético via OpenAI GPT-4o-mini.

    Requer OPENAI_API_KEY válida no ambiente.
    """
    try:
        from openai import OpenAI
        from src.config import OPENAI_MODEL

        client = OpenAI(api_key=api_key)

        system_prompt = (
            "Você é um médico clínico experiente. Dado um relato informal de um paciente, "
            "redija um breve caso clínico em linguagem técnica médica (máximo 150 palavras), "
            "usando terminologia, jargões clínicos e hipóteses diagnósticas precisas. "
            "Não inclua tratamento, apenas anamnese, exame físico e hipóteses."
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Relato do paciente: {query}"},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[HyDE] Falha na OpenAI API ({e}). Usando fallback offline.")
        return generate_hypothetical_document_offline(query)


def generate_hypothetical_document(query: str) -> str:
    """Ponto de entrada do HyDE — seleciona modo automático.

    Usa OpenAI se OPENAI_API_KEY estiver definida, caso contrário usa fallback offline.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if api_key:
        print("[HyDE] Usando OpenAI API para geração do documento hipotético.")
        doc = generate_hypothetical_document_openai(query, api_key)
    else:
        print("[HyDE] Modo offline — usando mapeamento semântico local.")
        doc = generate_hypothetical_document_offline(query)

    return doc
