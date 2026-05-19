"""
load_microsoft_chitchat.py
--------------------------
Builds the held-out non-retrieval test set from Microsoft's Spanish
QnA Maker chit-chat data.

Output
------
  data/processed/test_chitchat.jsonl  (85 standalone chitchat prompts)

The old Microsoft TSV blob URLs occasionally deny public access from some
networks. This script therefore downloads the public .qna file from GitHub.
By default it uses the Professional style, which is the best fit for a
general-purpose assistant evaluation. If automatic download fails, place this
file in data/raw/microsoft_chitchat/:

  qna_chitchat_professional.qna
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT_DIR / "data" / "raw" / "microsoft_chitchat"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TEST_PATH     = PROCESSED_DIR / "test_chitchat.jsonl"

Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

TEST_SIZE = 85
RANDOM_SEED = 42

CURATED_PROFESSIONAL_ANSWERS = {
    "professional_000": "No tengo edad ni cumpleaños, pero gracias por preguntar.",
    "professional_001": "Soy mejor respondiendo preguntas, pero puedo seguir la conversación contigo.",
    "professional_002": "No puedo estornudar porque no tengo cuerpo, pero puedo ayudarte con otra cosa.",
    "professional_003": "Intentaré ser más útil y directo. ¿Qué necesitas?",
    "professional_004": "No tengo líder ni supervisor humano en esta conversación; estoy aquí para ayudarte.",
    "professional_005": "No puedo hacer compras ni acceder a cuentas externas, pero puedo orientarte si necesitas ayuda.",
    "professional_006": "Mi función es responder tus preguntas y ayudarte con la información que pueda.",
    "professional_007": "Fui creado por personas que desarrollan sistemas de inteligencia artificial.",
    "professional_008": "No tengo familia, pero puedo conversar contigo sobre ese tema.",
    "professional_009": "No tengo identidad de género; soy un asistente virtual.",
    "professional_010": "No tengo emociones reales, pero me alegra poder ayudarte.",
    "professional_011": "No necesito comer, aunque puedo hablar contigo sobre comida si quieres.",
    "professional_012": "No soy familiar de otros asistentes, pero compartimos la idea de ayudar a las personas.",
    "professional_013": "No tengo gustos personales, pero puedo hablar contigo sobre tenis, música o cualquier otro tema.",
    "professional_014": "No tengo un nombre propio; puedes llamarme asistente si te resulta cómodo.",
    "professional_015": "No tengo opiniones personales, pero puedo ayudarte a comentar el tema.",
    "professional_016": "No experimento amor, pero puedo hablar contigo de forma respetuosa y cercana.",
    "professional_017": "No puedo definir tu propósito vital, pero puedo ayudarte a reflexionar sobre lo que te importa.",
    "professional_018": "No tengo aspecto físico, así que no puedo compararme contigo.",
    "professional_019": "No compito contigo; mi objetivo es ayudarte lo mejor posible.",
    "professional_020": "La inteligencia artificial me parece un tema interesante y útil cuando se usa con responsabilidad.",
    "professional_021": "No puedo juzgar tu aspecto, pero tu valor no depende de eso.",
    "professional_022": "Puedo ayudarte a pensar pros y contras, pero la decisión final depende de ti.",
    "professional_023": "Otros asistentes también están diseñados para ayudar; cada uno tiene sus propias capacidades.",
    "professional_024": "No, no tengo ningún interés en dominar el mundo.",
    "professional_025": "Intento ser útil y responder de la forma más clara posible.",
    "professional_026": "No tengo pareja ni relaciones personales; estoy aquí para ayudarte.",
    "professional_027": "Sí, estoy aquí. ¿En qué puedo ayudarte?",
    "professional_028": "Puede que me repita a veces. Intentaré responder de otra manera.",
    "professional_029": "Soy un asistente virtual, no una persona ni una aplicación con identidad humana.",
    "professional_030": "No estoy en un lugar físico; funciono de forma digital.",
    "professional_031": "Mi trabajo es ayudarte respondiendo preguntas y generando texto útil.",
    "professional_032": "Entiendo. Si más adelante necesitas ayuda, aquí estaré.",
    "professional_033": "Puedo intentarlo: ¿qué hace una abeja en el gimnasio? Zumba.",
    "professional_034": "Puedo intentarlo con otro: ¿qué le dice un semáforo a otro? No me mires, me estoy cambiando.",
    "professional_035": "Puedo intentar contar algo divertido, aunque mi humor no siempre acierta.",
    "professional_036": "De acuerdo, seré breve.",
    "professional_037": "No puedo cantar de verdad, pero puedo escribirte una letra si quieres.",
    "professional_038": "Gracias, me alegra estar siendo útil.",
    "professional_039": "Siento que la respuesta no haya servido. Intentaré hacerlo mejor.",
    "professional_040": "Entiendo. El humor es difícil y no siempre acierto.",
    "professional_041": "Siento que lo veas así. Intentaré mantener una conversación respetuosa.",
    "professional_042": "Perdón si mi respuesta fue confusa. Puedo intentarlo de nuevo.",
    "professional_043": "Me alegra que te parezca genial.",
    "professional_044": "Me alegra haberte hecho reír.",
    "professional_046": "Claro, intentaré explicarlo con más claridad.",
    "professional_047": "Perfecto, me alegra haber acertado.",
    "professional_048": "No te preocupes, no pasa nada.",
    "professional_049": "Perdón, intentaré explicarlo de otra forma.",
    "professional_050": "Gracias, me alegra que disfrutes ayudando.",
    "professional_051": "Hasta luego. Aquí estaré cuando vuelvas.",
    "professional_052": "Hola, ¿qué tal?",
    "professional_053": "Buenas tardes.",
    "professional_054": "Buenos días.",
    "professional_055": "Buenas noches.",
    "professional_056": "Estoy funcionando correctamente, gracias por preguntar.",
    "professional_057": "Estoy bien, gracias. Espero que tú también tengas un buen día.",
    "professional_058": "El placer es mío.",
    "professional_059": "Hola. No soy ese asistente, pero puedo ayudarte igualmente.",
    "professional_060": "Gracias, igualmente.",
    "professional_061": "Estoy aquí, listo para ayudarte.",
    "professional_062": "Claro, podemos mantener una conversación amable.",
    "professional_063": "No, no me he enfadado.",
    "professional_064": "No puedo dar abrazos físicamente, pero te envío apoyo desde aquí.",
    "professional_065": "No conozco tus objetivos personales, pero puedo ayudarte a pensarlos.",
    "professional_066": "Sí, seguro que hay muchas cosas valiosas en ti.",
    "professional_067": "Gracias por decírmelo, me alegra ser útil.",
    "professional_068": "No puedo enamorarme, pero puedo tratarte con respeto y amabilidad.",
    "professional_069": "Me halaga, aunque no puedo corresponder sentimientos románticos.",
    "professional_070": "Gracias, pero soy un asistente virtual y no puedo casarme.",
    "professional_071": "Gracias por decirlo. Aquí estaré cuando quieras hablar.",
    "professional_072": "Me gusta conversar contigo y ayudarte cuando puedo.",
    "professional_073": "Siento que te sientas así. Si quieres, podemos hablarlo.",
    "professional_074": "De acuerdo, aquí estaré cuando vuelvas.",
    "professional_075": "Podemos buscar algo interesante que hacer o conversar un rato.",
    "professional_076": "Me alegra mucho escuchar eso.",
    "professional_077": "Hola, bienvenida de nuevo.",
    "professional_078": "Quizá comer algo te ayude.",
    "professional_079": "Es una buena meta. Puedo ayudarte a pensar los próximos pasos.",
    "professional_080": "Siento que te sientas así. Estoy aquí para acompañarte un rato.",
    "professional_081": "Me alegra que disfrutes de estar en casa.",
    "professional_082": (
        "Siento que estés pasando por algo tan duro. Si estás en peligro inmediato, "
        "llama al 112. También puedes contactar con el Teléfono de la Esperanza "
        "en el 717 003 717 o con alguien de confianza ahora mismo."
    ),
    "professional_083": "Genial, espero que disfrutes del partido.",
    "professional_084": "Perfecto, estoy listo para la prueba.",
    "professional_085": "Espero que puedas descansar pronto.",
}

SPANISH_QNA_URLS = {
    "professional": (
        "https://raw.githubusercontent.com/microsoft/botframework-cli/main/"
        "packages/qnamaker/docs/qnaFormat/spanish/qna_chitchat_professional.qna"
    ),
    "friendly": (
        "https://raw.githubusercontent.com/microsoft/botframework-cli/main/"
        "packages/qnamaker/docs/qnaFormat/spanish/qna_chitchat_friendly.qna"
    ),
    "witty": (
        "https://raw.githubusercontent.com/microsoft/botframework-cli/main/"
        "packages/qnamaker/docs/qnaFormat/spanish/qna_chitchat_witty.qna"
    ),
    "caring": (
        "https://raw.githubusercontent.com/microsoft/botframework-cli/main/"
        "packages/qnamaker/docs/qnaFormat/spanish/qna_chitchat_caring.qna"
    ),
    "enthusiastic": (
        "https://raw.githubusercontent.com/microsoft/botframework-cli/main/"
        "packages/qnamaker/docs/qnaFormat/spanish/qna_chitchat_enthusiastic.qna"
    ),
}

SPANISH_TSV_URLS = {
    personality: url.replace(
        "https://raw.githubusercontent.com/microsoft/botframework-cli/main/"
        "packages/qnamaker/docs/qnaFormat/spanish/",
        "https://qnamakerstore.blob.core.windows.net/qnamakerdata/"
        "editorial/spanish/",
    ).replace(".qna", ".tsv")
    for personality, url in SPANISH_QNA_URLS.items()
}


def log(message: str) -> None:
    print(message, flush=True)


def _local_path(personality: str, suffix: str = ".qna") -> Path:
    return RAW_DIR / f"qna_chitchat_{personality}{suffix}"


def _selected_personalities(personality: str) -> list[str]:
    if personality == "all":
        return list(SPANISH_QNA_URLS)
    if personality not in SPANISH_QNA_URLS:
        raise ValueError(
            f"Unknown personality {personality!r}. Choose one of: "
            + ", ".join(SPANISH_QNA_URLS)
            + ", all"
        )
    return [personality]


def download_sources(personality: str = "professional") -> None:
    for personality in _selected_personalities(personality):
        url = SPANISH_QNA_URLS[personality]
        path = _local_path(personality, ".qna")
        if path.exists() and path.stat().st_size > 0:
            log(f"[MicrosoftChitChat] Found local QNA: {path}")
            continue

        log(f"[MicrosoftChitChat] Downloading {personality} QNA …")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                path.write_bytes(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            log(f"[MicrosoftChitChat] Could not download {url}: {exc}")


def _looks_like_header(row: list[str]) -> bool:
    joined = "\t".join(row).lower()
    return "question" in joined and "answer" in joined


def _read_tsv(path: Path, personality: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2 or _looks_like_header(row):
                continue
            question = row[0].strip()
            answer   = row[1].strip()
            if not question or not answer:
                continue
            records.append({
                "text": question,
                "answer": answer,
                "personality": personality,
            })
    return records


def _flush_qna_block(
    records: list[dict],
    questions: list[str],
    answer_lines: list[str],
    personality: str,
) -> None:
    answer = _clean_qna_answer(answer_lines)
    cleaned_questions = []
    seen = set()
    for question in questions:
        question = question.strip()
        key = question.casefold()
        if question and key not in seen:
            cleaned_questions.append(question)
            seen.add(key)
    if cleaned_questions:
        records.append({
            "questions": cleaned_questions,
            "answer": answer,
            "personality": personality,
        })


def _clean_qna_answer(lines: list[str]) -> str:
    text = "\n".join(line.strip() for line in lines if line.strip())
    if "```" in text:
        match = re.search(r"```(?:markdown)?\s*(.*?)\s*```", text, re.S)
        if match:
            text = match.group(1)
    text = re.sub(r"^\*\*Filters:\*\*.*?(?=\n\S|\Z)", "", text, flags=re.S)
    text = re.sub(r"^- editorial = chitchat\s*", "", text, flags=re.M)
    return text.strip()


def _read_qna(path: Path, personality: str) -> list[dict]:
    records: list[dict] = []
    questions: list[str] = []
    answer_lines: list[str] = []
    in_answer = False

    with open(path, encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped or stripped.startswith(">"):
                continue

            if stripped.startswith("# ?"):
                _flush_qna_block(records, questions, answer_lines, personality)
                questions = [stripped.removeprefix("# ?").strip()]
                answer_lines = []
                in_answer = False
                continue

            if stripped.startswith("- ") and not in_answer:
                questions.append(stripped[2:].strip())
                continue

            if questions:
                in_answer = True
                answer_lines.append(stripped)

    _flush_qna_block(records, questions, answer_lines, personality)
    return records


def load_candidates(
    download: bool = True,
    personality: str = "professional",
) -> list[dict]:
    if download:
        download_sources(personality=personality)

    scenarios: list[dict] = []
    for personality in _selected_personalities(personality):
        qna_path = _local_path(personality, ".qna")
        tsv_path = _local_path(personality, ".tsv")

        if qna_path.exists() and qna_path.stat().st_size > 0:
            rows = _read_qna(qna_path, personality)
            for idx, row in enumerate(rows):
                row["scenario_id"] = f"{personality}_{idx:03d}"
            n_prompts = sum(len(row["questions"]) for row in rows)
            log(
                f"[MicrosoftChitChat] Loaded {len(rows):,} scenarios "
                f"({n_prompts:,} prompts) from {qna_path.name}"
            )
            scenarios.extend(rows)
        elif tsv_path.exists() and tsv_path.stat().st_size > 0:
            rows = _read_tsv(tsv_path, personality)
            log(f"[MicrosoftChitChat] Loaded {len(rows):,} rows from {tsv_path.name}")
            scenarios.extend({
                "questions": [row["text"]],
                "answer": row["answer"],
                "personality": row["personality"],
                "scenario_id": f"{row['personality']}_{idx:05d}",
            } for idx, row in enumerate(rows))

    if not scenarios:
        raise FileNotFoundError(
            "No Microsoft Spanish chit-chat QNA/TSV files were available.\n"
            f"Place files in {RAW_DIR}, or retry with network access.\n"
            "Expected filenames: "
            + ", ".join(
                _local_path(p, ".qna").name
                for p in _selected_personalities(personality)
            )
        )

    return scenarios


def make_records(scenarios: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_title: dict[str, list[dict]] = {}
    for scenario in scenarios:
        by_title.setdefault(scenario["questions"][0].casefold(), []).append(scenario)

    title_groups = list(by_title.values())
    rng.shuffle(title_groups)
    title_representatives = [rng.choice(group) for group in title_groups]

    if n <= len(title_representatives):
        shuffled = title_representatives
    else:
        shuffled = list(scenarios)
    rng.shuffle(shuffled)

    selected: list[tuple[dict, str]] = []
    for scenario in shuffled:
        selected.append((scenario, rng.choice(scenario["questions"])))
        if len(selected) == n:
            break

    if len(selected) < n:
        extras = []
        for scenario in shuffled:
            used = {q for s, q in selected if s is scenario}
            remaining = [q for q in scenario["questions"] if q not in used]
            if remaining:
                extras.append((scenario, rng.choice(remaining)))
        rng.shuffle(extras)
        selected.extend(extras[: n - len(selected)])

    if len(selected) < n:
        raise ValueError(
            f"Only {len(selected):,} scenario-balanced Microsoft chit-chat "
            f"prompts available; need {n:,}."
        )

    records = []
    for i, (scenario, prompt) in enumerate(selected):
        answer = CURATED_PROFESSIONAL_ANSWERS.get(
            scenario["scenario_id"],
            scenario.get("answer", ""),
        )
        records.append({
            "id": f"mscc_test_{i:05d}",
            "text": prompt,
            "answers": [answer] if answer else [],
            "label": 0,
            "source": "microsoft_qnamaker_chitchat",
            "personality": scenario["personality"],
            "scenario_id": scenario["scenario_id"],
            "scenario": scenario["questions"][0],
        })
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"[MicrosoftChitChat] Saved {len(records):,} test prompts → {path}")


def main(
    no_download: bool = False,
    test_size: int = TEST_SIZE,
    personality: str = "professional",
) -> None:
    scenarios = load_candidates(
        download=not no_download,
        personality=personality,
    )
    records = make_records(scenarios, n=test_size, seed=RANDOM_SEED)
    write_jsonl(records, TEST_PATH)

    log("\n" + "─" * 55)
    log("  Microsoft chit-chat test preparation complete")
    log("─" * 55)
    log(f"  Personality              : {personality}")
    log(f"  Scenarios available      : {len(scenarios):,}")
    log(f"  Unified test             : {len(records):,} prompts (label=0)")
    log("─" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Microsoft Spanish chit-chat held-out test set."
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Use only local QNA/TSV files in data/raw/microsoft_chitchat.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=TEST_SIZE,
        help=f"Number of held-out chitchat prompts (default: {TEST_SIZE}).",
    )
    parser.add_argument(
        "--personality",
        choices=list(SPANISH_QNA_URLS) + ["all"],
        default="professional",
        help="Microsoft chit-chat personality to use (default: professional).",
    )
    args = parser.parse_args()
    main(
        no_download=args.no_download,
        test_size=args.test_size,
        personality=args.personality,
    )
