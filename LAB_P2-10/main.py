"""
LAB P2-10 — O Pipeline Definitivo: RAG + QLoRA + Otimização de Inferência na GPU

HealthTech — Sistema de Geração de Relatórios Médicos Automatizados
Missão: resolver OOM na GPU combinando QLoRA 4-bit + KV Cache + atenção otimizada

Passo 1: Ingestão eficiente — modelo carregado em 4-bits (QLoRA)
Passo 2: Simulação do RAG massivo — 5 capítulos médicos (~12.000 tokens brutos)
Passo 3: Geração SEM KV Cache — baseline com recálculo O(n²) por token
Passo 4: Geração COM KV Cache + atenção otimizada — pipeline otimizado
         Hierarquia de fallback: FlashAttention-2 → SDPA (PyTorch) → eager
         FA2 exige GPU Ampere+ (A100/3090+). T4 do Colab free → cai em SDPA.
Passo 5: Relatório comparativo de métricas
"""

from __future__ import annotations

import gc
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 100
TARGET_CONTEXT_WORDS = 8_500  # ~11.000 tokens antes de truncagem

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_GPU = DEVICE == "cuda"

print(f"\n{'=' * 60}")
print("  LAB P2-10 — Pipeline Definitivo: RAG + QLoRA + GPU Opt.")
print(f"  Dispositivo : {DEVICE.upper()}")
if HAS_GPU:
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    print(f"  GPU         : {gpu_name} ({total_vram:.0f} MB VRAM total)")
else:
    print("  AVISO: GPU nao detectada. Metricas de VRAM serao N/A.")
    print("  Para resultados reais, execute no Google Colab (T4/A100).")
print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────
# Utilitários de memória VRAM
# ─────────────────────────────────────────────────────────────
def vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024 ** 2 if HAS_GPU else 0.0


def peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024 ** 2 if HAS_GPU else 0.0


def reset_peak() -> None:
    if HAS_GPU:
        torch.cuda.reset_peak_memory_stats()


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ─────────────────────────────────────────────────────────────
# Passo 2 — Contexto médico simulado (5 capítulos RAG)
# ─────────────────────────────────────────────────────────────
MEDICAL_CHAPTERS = [
    """\
CAPÍTULO 1 — INSUFICIÊNCIA CARDÍACA CONGESTIVA (ICC)

1.1 Definição e Epidemiologia
A insuficiência cardíaca congestiva é uma síndrome clínica complexa caracterizada pela incapacidade
do coração de bombear sangue em quantidade suficiente para atender às demandas metabólicas dos
tecidos, ou de fazê-lo apenas com pressões de enchimento elevadas. Afeta mais de 64 milhões de
pessoas mundialmente, com prevalência crescente em populações idosas. No Brasil, representa a
principal causa de hospitalização em pacientes acima de 60 anos.

1.2 Fisiopatologia
A disfunção miocárdica primária deflagra uma cascata de mecanismos compensatórios: ativação do
sistema renina-angiotensina-aldosterona (SRAA), hiperatividade simpática e liberação de peptídeos
natriuréticos (BNP e NT-proBNP). A retenção hidrossalina provoca sobrecarga de volume, elevação das
pressões de enchimento ventricular esquerdo e congestão pulmonar. A ICC com fração de ejeção
reduzida (ICFEr, FEVE < 40%) distingue-se da ICC com fração de ejeção preservada (ICFEp, FEVE ≥
50%) por mecanismos fisiopatológicos distintos.

1.3 Diagnóstico
Os critérios de Framingham permanecem úteis: dois critérios maiores ou um maior e dois menores.
Critérios maiores incluem dispneia paroxística noturna, ortopneia, estertores pulmonares, edema agudo
de pulmão, cardiomegalia à radiografia torácica e ritmo de galope (B3). Critérios menores: dispneia
aos esforços, tosse noturna, taquicardia (FC > 120 bpm), edema bilateral de tornozelos e hepatomegalia.
Exames essenciais: ecocardiograma transtorácico (padrão-ouro), dosagem de BNP.

1.4 Tratamento Farmacológico
Três pilares com evidência classe I: (1) Inibidores da ECA ou BRA-II — reduzem pós-carga e inibem
remodelamento miocárdico; (2) Betabloqueadores (carvedilol, metoprolol succinato) — controle da
taquicardia reflexa; (3) Antagonistas da aldosterona — redução de fibrose. A associação
sacubitril/valsartana (ARNI) demonstrou superioridade no estudo PARADIGM-HF. Inibidores de SGLT-2
representam a mais recente adição com benefício cardiorrenometabólico comprovado.

1.5 Prognóstico
A classificação NYHA estratifica sintomas: Classe I a IV. Mortalidade em 5 anos: 50% em pacientes
com ICC Classe NYHA III-IV, comparável a neoplasias malignas. Biomarcadores prognósticos: BNP > 400
pg/mL, troponina elevada, hiponatremia (Na+ < 130 mEq/L) conferem pior prognóstico independente.\
""",

    """\
CAPÍTULO 2 — DIABETES MELLITUS TIPO 2 (DM2)

2.1 Epidemiologia Global
O Diabetes Mellitus tipo 2 constitui pandemia do século XXI, afetando 537 milhões de adultos
globalmente (IDF, 2021), com projeção de 783 milhões para 2045. No Brasil, a prevalência estimada é
de 16,8 milhões de pessoas, tornando o país o sexto em número absoluto de diabéticos. Representa a
principal causa de cegueira adquirida, amputação de membros inferiores e insuficiência renal crônica.

2.2 Fisiopatologia: O Octeto Ominoso de DeFronzo
Múltiplos defeitos orgânicos simultâneos: (1) Resistência à insulina hepática — aumento da produção
hepática de glicose; (2) Disfunção de células beta — redução progressiva da secreção insulínica; (3)
Resistência muscular; (4) Lipólise aumentada; (5) Comprometimento do efeito incretínico (GLP-1); (6)
Hiperglucagonemia; (7) Reabsorção renal aumentada (SGLT-2); (8) Disfunção neurotransmissora central.
A progressiva falência das células beta — à taxa de 4-5% ao ano — exige escalonamento terapêutico.

2.3 Critérios Diagnósticos (ADA 2023)
Glicemia de jejum ≥ 126 mg/dL (em duas ocasiões); Glicemia pós-sobrecarga de 75g ≥ 200 mg/dL aos
120 minutos (TOTG); HbA1c ≥ 6,5% (método NGSP); Glicemia casual ≥ 200 mg/dL com sintomas clássicos.
O pré-diabetes: glicemia de jejum 100-125 mg/dL ou HbA1c 5,7-6,4%, risco de progressão 5-10%/ano.

2.4 Estratégia Terapêutica
Metformina: primeira linha (redução HbA1c 1,0-1,5%). Com DCV estabelecida → iGLP-1 (semaglutida) ou
iSGLT-2 (empagliflozina). Com ICC ou DRC → iSGLT-2 preferencial. Com necessidade de emagrecimento
→ iGLP-1. Insulinoterapia basal (glargina, degludeca) indicada quando HbA1c > 10%.

2.5 Complicações
Retinopatia: fundoscopia anual. Nefropatia: microalbuminúria precede macroalbuminúria; iSGLT-2 e
iSRAA retardam progressão. Neuropatia: distal simétrica (mais prevalente), autonômica, focal.
Síndrome do pé diabético: amputação 15x mais frequente que em não diabéticos.\
""",

    """\
CAPÍTULO 3 — HIPERTENSÃO ARTERIAL SISTÊMICA (HAS)

3.1 Definição e Magnitude
HAS define-se como pressão arterial sistólica ≥ 140 mmHg e/ou diastólica ≥ 90 mmHg em medições
repetidas padronizadas. Acomete 1,28 bilhão de adultos globalmente (OMS, 2023), sendo o principal
fator de risco modificável para doenças cardiovasculares, cerebrovasculares e renais. No Brasil,
prevalência de 38,1% entre adultos, com controle adequado em menos de 30% dos tratados.

3.2 Classificação Pressórica (7ª Diretriz Brasileira de HAS, 2016)
Pressão ótima: < 120/80 mmHg. Pré-hipertensão: 130-139/85-89 mmHg. HAS estágio 1: 140-159/90-99
mmHg. HAS estágio 2: 160-179/100-109 mmHg. HAS estágio 3: ≥ 180/110 mmHg. HAS sistólica isolada:
PAS ≥ 140 com PAD < 90 mmHg (comum em idosos por rigidez aórtica).

3.3 Investigação de Lesão de Órgão-Alvo
Coração: ECG, ecocardiograma (massa VE indexada > 115 g/m² em homens indica HVE). Rim:
microalbuminúria, creatinina, TFG. Cérebro: RM (leucoaraiose, lacunas isquêmicas). Retina: fundoscopia
— classificação Keith-Wagener-Barker: grau I (espessamento arteriolar), grau II (cruzamento AV
patológico), grau III (exsudatos/hemorragias), grau IV (papiledema — emergência hipertensiva).

3.4 Terapêutica Anti-Hipertensiva
Cinco classes com evidência: (1) Diuréticos tiazídicos (clortalidona — maior evidência no ALLHAT);
(2) IECA (enalapril, ramipril); (3) BRA-II (losartana, valsartana); (4) BCC diidropiridínicos
(anlodipino); (5) Betabloqueadores (carvedilol — indicação primária em ICC, pós-IAM). Protocolo
combinado: IECA/BRA + BCC + tiazídico. Meta: < 130/80 mmHg (ACC/AHA 2017).

3.5 Crises Hipertensivas
Urgência: PA ≥ 180/120 mmHg sem lesão aguda; tratamento oral, redução gradual em 24-48h (captopril
SL). Emergência: elevação pressórica com lesão aguda (EAP, dissecção aórtica, encefalopatia,
eclâmpsia); nitroprussiato IV, nicardipina IV — redução de 20-25% da PAM na primeira hora.\
""",

    """\
CAPÍTULO 4 — DOENÇA PULMONAR OBSTRUTIVA CRÔNICA (DPOC)

4.1 Definição e Impacto Global
DPOC caracteriza-se por obstrução ao fluxo aéreo persistente e progressiva relacionada ao tabagismo
e exposição a partículas nocivas. Afeta 384 milhões de pessoas (prevalência 11,7%), sendo a terceira
causa de morte no mundo (OMS, 2023). No Brasil, subdiagnóstico superior a 70% pela subutilização
da espirometria.

4.2 Fisiopatologia e Fenótipos
Tabagismo deflagra inflamação crônica das vias aéreas por neutrófilos, macrófagos e linfócitos CD8+.
IL-8, TNF-α e leucotrieno B4 promovem hipertrofia glandular, hipersecreção, remodelamento fibroso e
destruição alveolar (desequilíbrio elastase/alfa-1-antitripsina). Fenótipos: "Pink Puffer" (enfisema —
magro, taquidispneico, barril torácico) vs "Blue Bloater" (bronquite crônica — obeso, cianótico,
hipercapnia, edema por cor pulmonale).

4.3 Diagnóstico Espirométrico (GOLD 2023)
VEF1/CVF < 0,70 pós-broncodilatador confirma DPOC. Estadiamento GOLD por VEF1 (% predito):
GOLD 1 (≥ 80%), GOLD 2 (50-79%), GOLD 3 (30-49%), GOLD 4 (< 30%). Estratificação ABCD: mMRC
(escala de dispneia) e CAT (escore 0-40). Grupo GOLD E: ≥ 2 exacerbações ou ≥ 1 hospitalização/ano.

4.4 Tratamento Farmacológico
LAMA (tiotrópio, umeclidínio) e LABA (salmeterol, indacaterol) constituem a base. Combinação
LAMA+LABA superior à monoterapia (estudos FLAME, IMPACT). Corticosteroide inalatório indicado em
ACO ou eosinofilia ≥ 300 células/μL — terapia tripla reduz exacerbações 25%. Roflumilaste indicado
em DPOC grave com bronquite crônica e exacerbações frequentes.

4.5 Reabilitação e Oxigenoterapia
Reabilitação pulmonar tem maior impacto funcional que qualquer intervenção farmacológica em DPOC
moderado-grave. Oxigenoterapia domiciliar (>15h/dia): indicada quando PaO2 ≤ 55 mmHg — único
tratamento além da cessação tabágica com evidência de redução de mortalidade (MRC 1981). VMNI nas
exacerbações com acidose respiratória: reduz mortalidade 50% e necessidade de VM invasiva.\
""",

    """\
CAPÍTULO 5 — ACIDENTE VASCULAR CEREBRAL ISQUÊMICO (AVCi)

5.1 Definição e Epidemiologia
AVCi é déficit neurológico focal de início súbito por oclusão arterial trombótica ou embólica.
Segunda causa de morte e principal causa de incapacidade adquirida em adultos no mundo (GBD 2019).
No Brasil, responde por 100.000 mortes anuais e é a primeira causa de óbito — maior que IAM. Custo
anual estimado: R$ 68 bilhões entre internações, reabilitação e perda de produtividade.

5.2 Fisiopatologia: Penumbra Isquêmica e Janela Terapêutica
Core isquêmico: necrose irreversível com CBF < 8-12 mL/100g/min; morte celular por depleção de ATP,
falência da bomba Na+/K+-ATPase, excitotoxicidade glutamatérgica. Penumbra isquêmica: zona com CBF
12-20 mL/100g/min — estruturalmente preservada, metabolicamente comprometida; alvo terapêutico.
Janela para trombólise IV: até 4,5 horas. Trombectomia mecânica: janela até 24h com mismatch
clinicorradiológico (DAWN/DEFUSE-3).

5.3 Diagnóstico: Escalas e Neuroimagem
FAST (Face, Arms, Speech, Time): sensibilidade 66%, especificidade 87%. NIHSS (0-42): < 4 leve, 5-15
moderado, > 20 grave. TC sem contraste: primeiro exame — exclui hemorragia (sensibilidade 98-100%).
RM-DWI: superior para core agudo (> 95% sensibilidade). TC de perfusão: quantifica core e penumbra,
guia trombectomia na janela ampliada.

5.4 Tratamento Agudo
rt-PA IV (alteplase 0,9 mg/kg, máx 90 mg): até 4,5h, NIHSS ≥ 4, PA ≤ 185/110 mmHg. Tenecteplase
(0,25 mg/kg): bolo único, aprovada FDA 2024, não inferior ao rt-PA. Trombectomia mecânica: oclusão
de grande vaso até 24h — recanalização 70-80% com técnica combinada stent retriever + aspiração.

5.5 Prevenção Secundária
AAS 100-300 mg/dia + clopidogrel por 21 dias em AIT e AVCi leve (POINT, CHANCE). Fibrilação atrial:
NOAC (apixabana, rivaroxabana) superiores à warfarina (ARISTOTLE, ROCKET-AF). Reabilitação: iniciar
em 24-48h — 60-70% dos sobreviventes recuperam marcha independente em 3 meses.\
""",
]


def generate_medical_context(target_words: int = TARGET_CONTEXT_WORDS) -> str:
    """Simula os 5 capítulos de manuais médicos recuperados pelo RAG."""
    base = "\n\n".join(MEDICAL_CHAPTERS)
    text = base
    while len(text.split()) < target_words:
        text += "\n\n" + base
    words = text.split()
    return " ".join(words[:target_words])


# ─────────────────────────────────────────────────────────────
# Passo 1 — Carregamento do modelo em 4-bits (QLoRA)
# ─────────────────────────────────────────────────────────────
def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model(
    use_flash_attention: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, float, str]:
    """Carrega TinyLlama em 4-bits (QLoRA).

    Hierarquia de atenção quando use_flash_attention=True:
      1. flash_attention_2 — requer GPU Ampere+ e pacote flash-attn
      2. sdpa             — torch.nn.functional.scaled_dot_product_attention (qualquer GPU)
      3. eager            — atenção padrão (CPU-safe)

    Retorna: (model, tokenizer, vram_usada_mb, impl_ativa)
    """
    attn_impl = "eager"
    fa_label = " + atenção otimizada" if use_flash_attention else ""
    section(f"PASSO 1 — Carregando modelo em 4-bits (QLoRA){fa_label}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if not HAS_GPU:
        print("  -> GPU nao disponivel. Carregando em CPU sem quantizacao 4-bit.")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        return model, tokenizer, 0.0, attn_impl

    reset_peak()
    vram_before = vram_mb()

    kwargs: dict = {
        "quantization_config": build_bnb_config(),
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    if use_flash_attention:
        # Tentativa 1: FlashAttention-2 (Ampere+: A100, RTX 3090+)
        kwargs["attn_implementation"] = "flash_attention_2"
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
            attn_impl = "flash_attention_2"
        except Exception as exc:
            print(f"  -> FlashAttention-2 indisponivel ({type(exc).__name__}).")
            print("     Requer GPU Ampere+ (A100, RTX 3090+). T4 do Colab free nao suporta.")
            print("     Tentando SDPA (Scaled Dot Product Attention do PyTorch)...")

            # Tentativa 2: SDPA — funciona em qualquer GPU com PyTorch >= 2.0
            kwargs["attn_implementation"] = "sdpa"
            try:
                model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
                attn_impl = "sdpa"
                print("  -> SDPA ativado com sucesso (resultado similar ao FA2).")
            except Exception as exc2:
                print(f"  -> SDPA tambem indisponivel ({type(exc2).__name__}). Usando eager.")
                kwargs.pop("attn_implementation")
                model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
                attn_impl = "eager"
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
        attn_impl = "eager"

    _attn_status = {
        "flash_attention_2": "ATIVO",
        "sdpa": "ATIVO via SDPA (fallback PyTorch — GPU qualquer)",
        "eager": "INATIVO (atenção padrão eager)",
    }
    vram_model = vram_mb() - vram_before
    print(f"  -> VRAM ocupada pelo modelo 4-bit : {vram_model:.1f} MB")
    print(f"  -> Atenção otimizada              : {_attn_status[attn_impl]}")
    return model, tokenizer, vram_model, attn_impl


# ─────────────────────────────────────────────────────────────
# Passos 3 e 4 — Benchmark de geração
# ─────────────────────────────────────────────────────────────
def benchmark(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    use_cache: bool,
    label: str,
) -> dict:
    """Executa geração de MAX_NEW_TOKENS e registra tempo + pico de VRAM."""
    section(f"Geração: {label}")
    model.config.use_cache = use_cache
    n_ctx = input_ids.shape[1]
    print(f"  use_cache          : {use_cache}")
    print(f"  Tokens de contexto : {n_ctx:,}")
    print(f"  Tokens a gerar     : {MAX_NEW_TOKENS}")

    if HAS_GPU:
        reset_peak()

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=use_cache,
        )
    elapsed = time.perf_counter() - t0
    peak = peak_vram_mb()
    tps = MAX_NEW_TOKENS / elapsed

    print(f"\n  Tempo total        : {elapsed:.2f} s")
    print(f"  Tokens por segundo : {tps:.1f} tok/s")
    if HAS_GPU:
        print(f"  Pico VRAM (GPU)    : {peak:.1f} MB")
    else:
        print("  Pico VRAM          : N/A (CPU)")

    return {"label": label, "time_s": elapsed, "tps": tps, "peak_vram_mb": peak}


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────
def main() -> None:
    # ── Passo 2: Gerar contexto massivo do RAG ────────────────
    section("PASSO 2 — Simulação do RAG Massivo (5 Capítulos Médicos)")
    ctx_text = generate_medical_context()
    n_words = len(ctx_text.split())
    print(f"  Texto gerado        : {n_words:,} palavras")
    print(f"  Tokens estimados    : ~{int(n_words * 1.3):,} tokens (antes de truncagem)")

    # ── Passo 1 + 3: Modelo padrão → benchmark SEM cache ─────
    model_std, tokenizer, vram_std, _ = load_model(use_flash_attention=False)

    # Tokenizar e truncar ao limite do modelo (2048 − MAX_NEW_TOKENS)
    max_ctx_len = model_std.config.max_position_embeddings - MAX_NEW_TOKENS
    enc = tokenizer(
        ctx_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_ctx_len,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    real_tokens = input_ids.shape[1]

    section("PASSO 2 (cont.) — Tokenização")
    print(f"  Janela maxima do modelo : {max_ctx_len:,} tokens")
    print(f"  Tokens apos truncagem   : {real_tokens:,} tokens")
    print(f"  (Em producao usaria modelo com janela de 32k+ como Llama-3)")

    # Passo 3: Geração sem KV Cache
    r_no_cache = benchmark(
        model_std, input_ids, use_cache=False,
        label="SEM KV Cache — O(n^2) por token (baseline lento)"
    )

    # Liberar modelo padrão da VRAM
    del model_std
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    # ── Passo 4: Modelo + atenção otimizada → benchmark COM cache ─
    model_opt, _, vram_opt, attn_impl_opt = load_model(use_flash_attention=True)
    _attn_display = {
        "flash_attention_2": "FlashAttention-2",
        "sdpa": "SDPA (PyTorch)",
        "eager": "Eager (padrão)",
    }
    attn_label = _attn_display.get(attn_impl_opt, attn_impl_opt)
    r_cache = benchmark(
        model_opt, input_ids.to(DEVICE), use_cache=True,
        label=f"COM KV Cache + {attn_label} (pipeline otimizado)"
    )
    del model_opt
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

    # ── Passo 5: Relatório comparativo ────────────────────────
    section("PASSO 5 — RELATÓRIO COMPARATIVO DE MÉTRICAS")

    speedup = r_no_cache["time_s"] / max(r_cache["time_s"], 0.001)
    vram_saved = r_no_cache["peak_vram_mb"] - r_cache["peak_vram_mb"]
    vram_pct = vram_saved / max(r_no_cache["peak_vram_mb"], 1.0) * 100

    col2_header = f"Cache+{attn_label}"[:13].center(13)
    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │          MÉTRICAS DE BENCHMARK — LAB P2-10               │
  ├──────────────────────────┬──────────────┬───────────────┤
  │ Métrica                  │  Sem Cache   │ {col2_header} │
  ├──────────────────────────┼──────────────┼───────────────┤
  │ VRAM carga 4-bit (MB)    │  {vram_std:>8.1f}    │  {vram_opt:>8.1f}   │
  │ Tokens de contexto       │  {real_tokens:>8,}    │  {real_tokens:>8,}   │
  │ Tempo de geração (s)     │  {r_no_cache['time_s']:>8.2f}    │  {r_cache['time_s']:>8.2f}   │
  │ Tokens / segundo         │  {r_no_cache['tps']:>8.1f}    │  {r_cache['tps']:>8.1f}   │
  │ Pico de VRAM (MB)        │  {r_no_cache['peak_vram_mb']:>8.1f}    │  {r_cache['peak_vram_mb']:>8.1f}   │
  ├──────────────────────────┼──────────────┼───────────────┤
  │ Speedup                  │      —       │  {speedup:>5.1f}x        │
  │ Reducao de VRAM          │      —       │  {vram_pct:>+5.1f}%        │
  └──────────────────────────┴──────────────┴───────────────┘

  Atenção usada na coluna otimizada : {attn_label}
  FA2 requer Ampere+ (A100/3090+). T4 (Colab free) usa SDPA como fallback.
  Em CPU apenas o tempo de geração é válido (VRAM = N/A).
    """)

    print("[LAB P2-10] Pipeline concluido com sucesso.\n")


if __name__ == "__main__":
    main()
