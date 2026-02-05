from __future__ import annotations

import csv
import io
import logging
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from flask import Flask

STOPWORDS_FR = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle",
    "en", "et", "eux", "il", "je", "la", "le", "leur", "lui", "ma", "mais", "me",
    "meme", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par",
    "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te",
    "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "c", "d", "j",
    "l", "m", "n", "s", "t", "y", "ete", "etre", "avoir", "faire", "stage", "stages",
}

WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass
class Fiche:
    numero: str
    intitule: str = ""
    abrege_libelle: str = ""
    abrege_intitule: str = ""
    niveau: str = ""
    niveau_intitule: str = ""
    actif: str = ""
    romes: List[str] = field(default_factory=list)
    rome_labels: List[str] = field(default_factory=list)
    formacodes: List[str] = field(default_factory=list)
    formacode_labels: List[str] = field(default_factory=list)
    blocs: List[str] = field(default_factory=list)

    def searchable_text(self) -> str:
        parts = [
            self.numero,
            self.intitule,
            self.abrege_libelle,
            self.abrege_intitule,
            self.niveau,
            self.niveau_intitule,
            " ".join(self.romes),
            " ".join(self.rome_labels),
            " ".join(self.formacodes),
            " ".join(self.formacode_labels),
            " ".join(self.blocs),
        ]
        return " ".join(p for p in parts if p)

    def text_by_category(self) -> Dict[str, str]:
        return {
            "job_title": " ".join(
                p
                for p in [
                    self.intitule,
                    self.abrege_libelle,
                    self.abrege_intitule,
                ]
                if p
            ),
            "domaine": " ".join(
                p
                for p in [
                    " ".join(self.rome_labels),
                    " ".join(self.formacode_labels),
                ]
                if p
            ),
            "niveau": " ".join(p for p in [self.niveau, self.niveau_intitule] if p),
            "competences": " ".join(p for p in self.blocs if p),
            "rome": " ".join(p for p in [" ".join(self.romes), " ".join(self.rome_labels)] if p),
            "formacode": " ".join(p for p in [" ".join(self.formacodes), " ".join(self.formacode_labels)] if p),
        }


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return [t.strip("'") for t in WORD_RE.findall(text) if t.strip("'")]


def split_keywords(query: str) -> List[str]:
    parts = [p.strip() for p in query.split(";")]
    return [p for p in parts if p]


def extract_keywords(query: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    return extract_keywords_with_vocab(query)


def extract_keywords_with_vocab(
    query: str,
    vocab: set[str] | None = None,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    if ";" in query:
        raw = split_keywords(query)
        normalized = []
        pairs = []
        for item in raw:
            norm = normalize_text(item)
            if not norm:
                continue
            if " " not in norm and len(norm) <= 2:
                continue
            if " " not in norm and norm in STOPWORDS_FR:
                continue
            normalized.append(norm)
            pairs.append((item, norm))
        return raw, normalized, pairs

    text = normalize_text(query)
    tokens = [
        t
        for t in tokenize(text)
        if len(t) > 2 and t not in STOPWORDS_FR and "'" not in t
    ]
    if vocab is not None:
        tokens = [t for t in tokens if t in vocab]
    unique = []
    seen = set()
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique, unique, [(t, t) for t in unique]


def read_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows


def register_template_filters(app: Flask) -> None:
    @app.template_filter("rncp_url")
    def rncp_url(code: str) -> str:
        if not code:
            return ""
        text = normalize_text(code)
        m = re.search(r"([a-z]+)[^a-z0-9]*([0-9]+)", text)
        if not m:
            return ""
        return f"https://www.francecompetences.fr/recherche/{m.group(1)}/{m.group(2)}/"


class Store:
    def __init__(self, data_dir: str, logger: logging.Logger | None = None) -> None:
        self.data_dir = data_dir
        self.logger = logger or logging.getLogger("app.store")
        self._loaded = False

        self.fiches: Dict[str, Fiche] = {}
        self.token_index: Dict[str, Counter] = {}
        self.search_text: Dict[str, str] = {}
        self.category_text: Dict[str, Dict[str, str]] = {}
        self.category_index: Dict[str, Dict[str, Counter]] = {}
        self.db_vocab: set[str] = set()
        self.domain_labels: Dict[str, str] = {}
        self.domain_to_skills: Dict[str, set[str]] = {}
        self.domain_tokens: Dict[str, List[str]] = {}
        self.skill_vocab: set[str] = set()
        self.category_vocab: Dict[str, set[str]] = {
            "job_title": set(),
            "domaine": set(),
            "competences": set(),
            "rome": set(),
            "formacode": set(),
        }

        self._nlp = None
        self._spacy_ready = False
        self._spacy_stopwords: set[str] = set()
        self._domain_lemma_tokens: Dict[str, List[str]] = {}

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.logger.info("Loading RNCP data from %s", self.data_dir)
        self.fiches = self._load_data()
        self._build_indices()
        self._loaded = True

    def _load_data(self) -> Dict[str, Fiche]:
        fiches: Dict[str, Fiche] = {}

        standard_path = os.path.join(self.data_dir, "export_fiches_CSV_Standard_2026_02_01.csv")
        for row in read_csv(standard_path):
            numero = row.get("Numero_Fiche", "").strip()
            if not numero:
                continue
            fiches[numero] = Fiche(
                numero=numero,
                intitule=row.get("Intitule", "").strip(),
                abrege_libelle=row.get("Abrege_Libelle", "").strip(),
                abrege_intitule=row.get("Abrege_Intitule", "").strip(),
                niveau=row.get("Nomenclature_Europe_Niveau", "").strip(),
                niveau_intitule=row.get("Nomenclature_Europe_Intitule", "").strip(),
                actif=row.get("Actif", "").strip(),
            )

        def ensure(numero: str) -> Fiche:
            if numero not in fiches:
                fiches[numero] = Fiche(numero=numero)
            return fiches[numero]

        rome_path = os.path.join(self.data_dir, "export_fiches_CSV_Rome_2026_02_01.csv")
        for row in read_csv(rome_path):
            numero = row.get("Numero_Fiche", "").strip()
            if not numero:
                continue
            fiche = ensure(numero)
            code = row.get("Codes_Rome_Code", "").strip()
            label = row.get("Codes_Rome_Libelle", "").strip()
            if code:
                fiche.romes.append(code)
            if label:
                fiche.rome_labels.append(label)

        formacode_path = os.path.join(self.data_dir, "export_fiches_CSV_Formacode_2026_02_01.csv")
        for row in read_csv(formacode_path):
            numero = row.get("Numero_Fiche", "").strip()
            if not numero:
                continue
            fiche = ensure(numero)
            code = row.get("Formacode_Code", "").strip()
            label = row.get("Formacode_Libelle", "").strip()
            if code:
                fiche.formacodes.append(code)
            if label:
                fiche.formacode_labels.append(label)

        blocs_path = os.path.join(self.data_dir, "export_fiches_CSV_Blocs_De_Compétences_2026_02_01.csv")
        for row in read_csv(blocs_path):
            numero = row.get("Numero_Fiche", "").strip()
            if not numero:
                continue
            fiche = ensure(numero)
            label = row.get("Bloc_Competences_Libelle", "").strip()
            if label:
                fiche.blocs.append(label)

        return fiches

    def _build_indices(self) -> None:
        for numero, fiche in self.fiches.items():
            text = normalize_text(fiche.searchable_text())
            self.search_text[numero] = text
            tokens = tokenize(text)
            self.token_index[numero] = Counter(tokens)
            self.db_vocab.update(tokens)

            per_cat = {}
            per_cat_index = {}
            for cat, cat_text in fiche.text_by_category().items():
                norm_text = normalize_text(cat_text)
                per_cat[cat] = norm_text
                per_cat_index[cat] = Counter(tokenize(norm_text))
            self.category_text[numero] = per_cat
            self.category_index[numero] = per_cat_index

            for label in fiche.rome_labels + fiche.formacode_labels:
                norm_label = normalize_text(label)
                if norm_label:
                    self.domain_labels[norm_label] = label
                    if norm_label not in self.domain_tokens:
                        self.domain_tokens[norm_label] = [
                            t for t in tokenize(norm_label) if len(t) > 2 and t not in STOPWORDS_FR
                        ]
                    self.domain_to_skills.setdefault(norm_label, set()).update(
                        {
                            t
                            for t in tokenize(" ".join(fiche.blocs))
                            if len(t) > 2 and t not in STOPWORDS_FR and "'" not in t
                        }
                    )
            for bloc in fiche.blocs:
                for token in tokenize(bloc):
                    if len(token) > 2 and token not in STOPWORDS_FR:
                        self.skill_vocab.add(token)
                        self.category_vocab["competences"].add(token)

            for token in tokenize(f"{fiche.intitule} {fiche.abrege_libelle} {fiche.abrege_intitule}"):
                if len(token) > 2 and token not in STOPWORDS_FR:
                    self.category_vocab["job_title"].add(token)
            for token in tokenize(" ".join(fiche.rome_labels)):
                if len(token) > 2 and token not in STOPWORDS_FR:
                    self.category_vocab["rome"].add(token)
                    self.category_vocab["domaine"].add(token)
            for token in tokenize(" ".join(fiche.formacode_labels)):
                if len(token) > 2 and token not in STOPWORDS_FR:
                    self.category_vocab["formacode"].add(token)
                    self.category_vocab["domaine"].add(token)

    def init_spacy(self) -> bool:
        if self._spacy_ready:
            return self._nlp is not None
        self._spacy_ready = True
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load("fr_core_news_sm")
            self._spacy_stopwords = {normalize_text(w) for w in self._nlp.Defaults.stop_words}
            self._spacy_stopwords.update(STOPWORDS_FR)
            for norm_label, original in self.domain_labels.items():
                doc = self._nlp(original)
                lemmas = [normalize_text(t.lemma_) for t in doc if t.is_alpha]
                self._domain_lemma_tokens[norm_label] = [
                    t for t in lemmas if len(t) > 2 and t not in self._spacy_stopwords
                ]
            return True
        except Exception as exc:
            self.logger.warning("Spacy unavailable: %s", exc)
            self._nlp = None
            return False

    @staticmethod
    def keyword_variants(keyword: str) -> List[str]:
        variants = {keyword}
        m = re.search(r"\\bniveau\\s*(\\d)\\b", keyword)
        if m:
            variants.add(f"niv{m.group(1)}")
        m = re.search(r"\\bniv\\s*(\\d)\\b", keyword)
        if m:
            variants.add(f"niveau {m.group(1)}")
        return list(variants)

    @staticmethod
    def normalize_niveau_input(value: str) -> str:
        trimmed = value.strip()
        if re.fullmatch(r"\\d", trimmed):
            return f"niveau {trimmed}"
        return trimmed

    @staticmethod
    def requested_niveau(value: str) -> int | None:
        if not value:
            return None
        text = normalize_text(value)
        m = re.search(r"(?:niv|niveau)\\s*(\\d)", text)
        if m:
            return int(m.group(1))
        if re.fullmatch(r"\\d", text):
            return int(text)
        return None

    def niveau_value(self, fiche: Fiche) -> int:
        candidates = [fiche.niveau, fiche.niveau_intitule]
        for item in candidates:
            if not item:
                continue
            text = normalize_text(item)
            m = re.search(r"(?:niv|niveau)\\s*(\\d)", text)
            if m:
                return int(m.group(1))
        return 999

    @staticmethod
    def actif_rank(fiche: Fiche) -> int:
        text = normalize_text(fiche.actif)
        if "active" in text or "actif" in text:
            return 0
        return 1

    def score_fiche(
        self,
        pairs_by_cat: Dict[str, List[Tuple[str, str]]],
        numero: str,
    ) -> Tuple[int, int, List[str], int]:
        text_by_cat = self.category_text[numero]
        index_by_cat = self.category_index[numero]
        matched = []
        score = 0
        expected = 0
        for cat, pairs in pairs_by_cat.items():
            if not pairs:
                continue
            expected += len(pairs)
            counter = index_by_cat.get(cat, Counter())
            text = text_by_cat.get(cat, "")
            for display, kw in pairs:
                variants = self.keyword_variants(kw)
                hit = False
                for v in variants:
                    if " " in v:
                        if v in text:
                            hit = True
                            score += len(tokenize(v)) or 1
                            break
                    else:
                        count = counter.get(v, 0)
                        if count > 0:
                            hit = True
                            score += count
                            break
                if hit:
                    matched.append(display)
        return len(matched), score, matched, expected

    def build_pairs_by_cat(
        self,
        query_by_cat: Dict[str, str],
    ) -> Tuple[Dict[str, List[Tuple[str, str]]], List[Tuple[str, str]], int | None]:
        niveau_limit = self.requested_niveau(query_by_cat.get("niveau", ""))
        pairs_by_cat: Dict[str, List[Tuple[str, str]]] = {}
        keywords_display: List[Tuple[str, str]] = []

        for cat, value in query_by_cat.items():
            if not value:
                pairs_by_cat[cat] = []
                continue
            if cat == "niveau":
                raw = split_keywords(value)
                keywords_display.extend([(cat, r) for r in raw])
                pairs_by_cat[cat] = []
                continue
            vocab = self.category_vocab.get(cat)
            raw, _, pairs = extract_keywords_with_vocab(value, vocab)
            keywords_display.extend([(cat, r) for r in raw])
            pairs_by_cat[cat] = pairs
        return pairs_by_cat, keywords_display, niveau_limit

    def search(
        self,
        query_by_cat: Dict[str, str],
        limit: int = 20,
    ) -> Tuple[List[Tuple[Fiche, int, List[str], int, int]], List[Tuple[str, str]]]:
        self.ensure_loaded()
        pairs_by_cat, keywords_display, niveau_limit = self.build_pairs_by_cat(query_by_cat)

        if not any(pairs_by_cat.values()):
            return [], []

        scored = []
        for numero in self.token_index.keys():
            fiche = self.fiches[numero]
            if niveau_limit is not None:
                fiche_niveau = self.niveau_value(fiche)
                if fiche_niveau > niveau_limit:
                    continue
            matched_count, score, matched, expected = self.score_fiche(pairs_by_cat, numero)
            if matched_count > 0:
                scored.append((fiche, matched_count, expected, score, matched))

        scored.sort(
            key=lambda x: (
                -(x[1] / max(x[2], 1)),
                -x[1],
                -x[3],
                self.actif_rank(x[0]),
                self.niveau_value(x[0]),
            )
        )

        results = []
        for fiche, matched_count, expected, score, matched in scored[:limit]:
            results.append((fiche, score, matched, matched_count, expected))
        return results, keywords_display

    def search_with_fallback(
        self,
        query_by_cat: Dict[str, str],
        limit: int = 20,
        fallback_limit: int = 10,
    ) -> Tuple[
        List[Tuple[Fiche, int, List[str], int, int]],
        List[Tuple[str, str]],
        List[Tuple[Fiche, int, List[str], int, int]],
    ]:
        self.ensure_loaded()
        pairs_by_cat, keywords_display, niveau_limit = self.build_pairs_by_cat(query_by_cat)
        if not any(pairs_by_cat.values()):
            return [], [], []

        results, _ = self.search(query_by_cat, limit=limit)
        result_ids = {r[0].numero for r in results}

        query_tokens = [kw for pairs in pairs_by_cat.values() for _, kw in pairs]
        if not query_tokens:
            return results, keywords_display, []

        fallback_scored = []
        for numero in self.token_index.keys():
            if numero in result_ids:
                continue
            fiche = self.fiches[numero]
            if niveau_limit is not None:
                fiche_niveau = self.niveau_value(fiche)
                if fiche_niveau > niveau_limit:
                    continue
            overlap = sum(self.token_index[numero].get(t, 0) for t in query_tokens)
            if overlap <= 0:
                continue
            fallback_scored.append((fiche, overlap))

        fallback_scored.sort(
            key=lambda x: (
                -x[1],
                self.actif_rank(x[0]),
                self.niveau_value(x[0]),
            )
        )

        fallback_results = []
        for fiche, overlap in fallback_scored[:fallback_limit]:
            fallback_results.append((fiche, overlap, [], 0, 0))

        return results, keywords_display, fallback_results

    def build_role_analysis(
        self,
        role: str,
        query_by_cat: Dict[str, str],
        results: List[Tuple[Fiche, int, List[str], int, int]],
    ) -> Dict[str, str]:
        analysis = {}
        if role == "ecole":
            if not results:
                analysis["status"] = "Aucune fiche RNCP compatible trouvée avec ces critères."
            else:
                analysis["status"] = "Fiches RNCP compatibles identifiées pour vérification pédagogique."
            if query_by_cat.get("niveau"):
                analysis["niveau"] = f"Niveau demandé : {query_by_cat.get('niveau')}"
            else:
                analysis["niveau"] = "Niveau non précisé par l’école."
        elif role == "employeur":
            if not results:
                analysis["status"] = "Aucune fiche RNCP évidente : l’offre semble trop spécifique ou incomplète."
            else:
                analysis["status"] = "Fiches RNCP compatibles avec les attentes exprimées."
            if query_by_cat.get("competences"):
                analysis["competences"] = "Compétences clés reconnues dans le référentiel."
            else:
                analysis["competences"] = "Compétences non précisées : l’offre gagnerait en clarté."
        return analysis

    @staticmethod
    def is_likely_name_line(line: str) -> bool:
        if not line:
            return False
        if re.search(r"[\\w.]+@[\\w.-]+\\.[a-z]{2,}", line.lower()):
            return True
        if re.search(r"\\+?\\d{2,}|\\d{2}\\s*\\d{2}", line):
            return True
        tokens = [t for t in line.replace("-", " ").split() if t]
        if 1 < len(tokens) <= 3:
            if all(t[:1].isupper() and t[1:].isalpha() for t in tokens if t.isalpha()):
                norm = normalize_text(line)
                if not re.search(r"\\b(cv|curriculum|profil|poste|intitule|fonction|metier)\\b", norm):
                    return True
        return False

    def pick_title_line(self, text: str) -> str:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return ""
        for line in lines[:10]:
            if self.is_likely_name_line(line):
                continue
            norm = normalize_text(line)
            if re.search(r"\\b(cv|curriculum|profil|poste|intitule|fonction|metier|recherche|objectif|resume)\\b", norm):
                return line
        for line in lines[:10]:
            if not self.is_likely_name_line(line):
                return line
        return lines[0]

    @staticmethod
    def extract_job_title_from_line(line: str) -> str:
        if not line:
            return ""
        cleaned = line.strip()
        m = re.search(r"[:\\-–]\\s*(.+)$", cleaned)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) >= 3:
                return candidate
        return cleaned

    @staticmethod
    def split_title_parts(line: str) -> List[str]:
        if not line:
            return []
        parts = re.split(r"[|•]|\\s[-–—]\\s", line)
        return [p.strip(" -–—|•") for p in parts if p.strip(" -–—|•")]

    def infer_domain_from_tokens(self, token_counts: Counter, title_tokens: set[str]) -> str:
        best_label = ""
        best_score = 0
        for norm_label, original in self.domain_labels.items():
            label_tokens = self.domain_tokens.get(norm_label, [])
            if not label_tokens:
                continue
            score = sum(token_counts.get(t, 0) for t in label_tokens)
            title_bonus = sum(1 for t in label_tokens if t in title_tokens)
            score += title_bonus * 3
            has_strong = any(len(t) >= 6 and token_counts.get(t, 0) > 0 for t in label_tokens)
            if score >= 2 or has_strong:
                if score > best_score:
                    best_score = score
                    best_label = original
        return best_label

    def build_focus_text(self, text: str, title_tokens: set[str]) -> str:
        if not text or not title_tokens:
            return ""
        lines = [l.strip() for l in text.splitlines()]
        focus_lines = []
        carry = 0
        for line in lines:
            if not line:
                carry = max(carry - 1, 0)
                continue
            norm = normalize_text(line)
            if re.search(r"\\b(competence|competences|skills|techniques|stack)\\b", norm):
                carry = 5
                focus_lines.append(line)
                continue
            if any(t in norm for t in title_tokens):
                carry = max(carry, 3)
                focus_lines.append(line)
                continue
            if carry > 0:
                focus_lines.append(line)
                carry -= 1
        return "\n".join(focus_lines)

    def parse_resume_text_heuristic(self, text: str) -> Dict[str, str]:
        norm = normalize_text(text)
        niveau = ""
        m = re.search(r"(?:niv|niveau)\\s*(\\d)", norm)
        if m:
            niveau = f"niveau {m.group(1)}"
        title_line = self.pick_title_line(text)
        title_parts = self.split_title_parts(title_line)
        job_title = self.extract_job_title_from_line(title_parts[0] if title_parts else title_line)
        domain_hint = title_parts[1] if len(title_parts) > 1 else ""
        title_norm = normalize_text(" ".join(p for p in [job_title, domain_hint] if p))
        cv_tokens = tokenize(norm)
        cv_counts = Counter(
            t for t in cv_tokens if len(t) > 2 and t not in STOPWORDS_FR and "'" not in t
        )
        title_tokens_raw = {t for t in tokenize(title_norm) if len(t) > 2 and t not in STOPWORDS_FR}
        title_tokens = {t for t in title_tokens_raw if t in self.category_vocab["job_title"]} or title_tokens_raw
        best_label = self.infer_domain_from_tokens(cv_counts, title_tokens)
        domain_hits = []
        if domain_hint:
            norm_hint = normalize_text(domain_hint)
            if norm_hint in self.domain_labels:
                domain_hits.append(self.domain_labels[norm_hint])
            else:
                domain_hits.append(domain_hint)
        elif best_label:
            domain_hits.append(best_label)

        focus_text = self.build_focus_text(text, title_tokens)
        focus_tokens = tokenize(normalize_text(focus_text)) if focus_text else cv_tokens
        filtered = [
            t
            for t in focus_tokens
            if len(t) > 2
            and t not in STOPWORDS_FR
            and "'" not in t
        ]
        generic_stop = {
            "projet", "projets", "mise", "les", "langue", "creation", "expert",
            "recherche", "apporter", "numerique", "sciences", "gestion", "developpement",
        }
        skill_vocab = self.skill_vocab
        if best_label:
            norm_domain = normalize_text(best_label)
            if norm_domain in self.domain_to_skills:
                skill_vocab = self.domain_to_skills[norm_domain]
        skill_hits = [t for t in filtered if t in skill_vocab and t not in generic_stop]
        skill_counts = Counter(skill_hits)
        skills = [w for w, _ in skill_counts.most_common(10)]

        return {
            "job_title": job_title,
            "domaine": " ; ".join(domain_hits),
            "competences": " ; ".join(skills),
            "niveau": niveau,
            "rome": "",
            "formacode": "",
        }

    def parse_resume_text_spacy(self, text: str) -> Dict[str, str]:
        norm = normalize_text(text)
        niveau = ""
        m = re.search(r"(?:niv|niveau)\\s*(\\d)", norm)
        if m:
            niveau = f"niveau {m.group(1)}"

        title_line = self.pick_title_line(text)
        title_parts = self.split_title_parts(title_line)
        job_title = self.extract_job_title_from_line(title_parts[0] if title_parts else title_line)
        domain_hint = title_parts[1] if len(title_parts) > 1 else ""
        title_norm = normalize_text(" ".join(p for p in [job_title, domain_hint] if p))

        if not self._nlp:
            return self.parse_resume_text_heuristic(text)

        doc = self._nlp(text[:12000])
        title_doc = self._nlp(title_norm) if title_norm else None

        def token_norm(token) -> str:
            return normalize_text(token)

        cv_tokens = []
        cv_counts = Counter()
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            surface = token_norm(token.text)
            lemma = token_norm(token.lemma_)
            for item in (lemma, surface):
                if not item or len(item) <= 2:
                    continue
                if item in self._spacy_stopwords:
                    continue
                cv_tokens.append(item)
                cv_counts[item] += 1

        title_tokens_raw = set()
        if title_doc:
            for token in title_doc:
                if token.is_space or token.is_punct:
                    continue
                item = token_norm(token.lemma_)
                if item and len(item) > 2 and item not in self._spacy_stopwords:
                    title_tokens_raw.add(item)
        title_tokens = {
            t for t in title_tokens_raw if t in self.category_vocab["job_title"]
        } or title_tokens_raw

        best_label = ""
        best_score = 0
        for norm_label, original in self.domain_labels.items():
            label_tokens = self._domain_lemma_tokens.get(norm_label, [])
            if not label_tokens:
                continue
            score = sum(cv_counts.get(t, 0) for t in label_tokens)
            title_bonus = sum(1 for t in label_tokens if t in title_tokens)
            score += title_bonus * 3
            has_strong = any(len(t) >= 6 and cv_counts.get(t, 0) > 0 for t in label_tokens)
            if score >= 2 or has_strong:
                if score > best_score:
                    best_score = score
                    best_label = original
        domain_hits = []
        if domain_hint:
            norm_hint = normalize_text(domain_hint)
            if norm_hint in self.domain_labels:
                domain_hits.append(self.domain_labels[norm_hint])
            else:
                domain_hits.append(domain_hint)
        elif best_label:
            domain_hits.append(best_label)

        focus_text = self.build_focus_text(text, title_tokens)
        if focus_text:
            focus_doc = self._nlp(focus_text[:12000])
            focus_tokens = []
            for token in focus_doc:
                if token.is_space or token.is_punct:
                    continue
                surface = token_norm(token.text)
                lemma = token_norm(token.lemma_)
                for item in (lemma, surface):
                    if not item or len(item) <= 2:
                        continue
                    if item in self._spacy_stopwords:
                        continue
                    focus_tokens.append(item)
        else:
            focus_tokens = cv_tokens

        generic_stop = {
            "projet", "projets", "mise", "les", "langue", "creation", "expert",
            "recherche", "apporter", "numerique", "sciences", "gestion", "developpement",
        }
        skill_vocab = self.skill_vocab
        if best_label:
            norm_domain = normalize_text(best_label)
            if norm_domain in self.domain_to_skills:
                skill_vocab = self.domain_to_skills[norm_domain]
        skill_hits = [
            t for t in focus_tokens if t in skill_vocab and t not in generic_stop
        ]
        skill_counts = Counter(skill_hits)
        skills = [w for w, _ in skill_counts.most_common(10)]

        return {
            "job_title": job_title,
            "domaine": " ; ".join(domain_hits),
            "competences": " ; ".join(skills),
            "niveau": niveau,
            "rome": "",
            "formacode": "",
        }

    def parse_resume_text(self, text: str) -> Dict[str, str]:
        self.ensure_loaded()
        if self.init_spacy():
            return self.parse_resume_text_spacy(text)
        return self.parse_resume_text_heuristic(text)


_STORE: Store | None = None


def get_store(app) -> Store:
    global _STORE
    data_dir = app.config["DATA_DIR"]
    if _STORE is None or _STORE.data_dir != data_dir:
        _STORE = Store(data_dir, app.logger)
    else:
        _STORE.logger = app.logger
    _STORE.ensure_loaded()
    return _STORE
