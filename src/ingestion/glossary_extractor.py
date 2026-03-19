"""Extracteur de glossaire pour le GRI.

Extrait les termes bilingues (FR/EN) avec leurs définitions
et références normatives (ISO/IEC/IEEE 15288:2023).
"""

import re
from pathlib import Path

from src.core.logging import get_logger
from src.ingestion.models import GlossaryEntry, ParsedSection, SectionType

log = get_logger(__name__)


class GRIGlossaryExtractor:
    """Extracteur de glossaire spécialisé pour le GRI.

    Détecte et extrait les définitions de termes ISO bilingues.
    """

    # Patterns pour détecter les définitions
    TERM_PATTERNS = [
        # Format: **Terme FR** (Term EN) : définition
        re.compile(
            r"\*\*([^*]+)\*\*\s*\(([^)]+)\)\s*[:\-–]\s*(.+)",
            re.IGNORECASE,
        ),
        # Format: Terme FR (Term EN) : définition
        re.compile(
            r"^([A-ZÀ-Ÿ][^(]+)\s*\(([^)]+)\)\s*[:\-–]\s*(.+)",
            re.MULTILINE,
        ),
        # Format: - Terme : définition
        re.compile(
            r"^[-•]\s*([^:]+)\s*:\s*(.+)",
            re.MULTILINE,
        ),
    ]

    # Pattern pour les références normatives
    STANDARD_REF_PATTERN = re.compile(
        r"(ISO[/\s]*(?:IEC)?[/\s]*(?:IEEE)?[/\s]*\d+[:\-]?\d*)",
        re.IGNORECASE,
    )

    # Termes connus du GRI (pour validation)
    KNOWN_TERMS = {
        "artefact",
        "artifact",
        "conops",
        "semp",
        "sow",
        "wbs",
        "trl",
        "mrl",
        "irl",
        "cdrl",
        "exigence",
        "requirement",
        "vérification",
        "verification",
        "validation",
        "intégration",
        "integration",
        "architecture",
        "système",
        "system",
        "jalon",
        "milestone",
        "phase",
        "cycle de vie",
        "lifecycle",
        "parties prenantes",
        "stakeholder",
        "traçabilité",
        "traceability",
        "baseline",
        "configuration",
        "risque",
        "risk",
    }

    def extract_from_sections(
        self,
        sections: list[ParsedSection],
    ) -> list[GlossaryEntry]:
        """Extrait le glossaire depuis les sections de type definition."""
        entries: list[GlossaryEntry] = []

        # Trouver les sections de glossaire/terminologie
        glossary_sections = [s for s in sections if s.section_type == SectionType.DEFINITION]

        log.info(
            "gri.glossary.sections_found",
            count=len(glossary_sections),
        )

        for section in glossary_sections:
            section_entries = self._extract_from_section(section)
            # Filtrer avec validation
            validated = [e for e in section_entries if self.validate_entry(e)]
            entries.extend(validated)

        # FALLBACK: Si aucune section DEFINITION trouvée, scanner tout le contenu
        if not entries:
            log.warning("gri.glossary.fallback_extraction")
            for section in sections:
                if self._looks_like_glossary_content(section.content):
                    section_entries = self._extract_from_section(section)
                    validated = [e for e in section_entries if self.validate_entry(e)]
                    entries.extend(validated)

        # Dédupliquer par terme FR
        unique_entries = self._deduplicate(entries)

        log.info(
            "gri.glossary.extracted",
            total=len(entries),
            unique=len(unique_entries),
        )

        return unique_entries

    def _looks_like_glossary_content(self, text: str) -> bool:
        """Heuristique pour détecter du contenu de type glossaire."""
        if not text:
            return False
        lines = text.split("\n")
        # Compter les lignes qui ressemblent à des définitions (terme : définition)
        colon_lines = sum(
            1 for line in lines if ":" in line and len(line.split(":")[0].strip()) < 50
        )
        return colon_lines >= 3  # Au moins 3 lignes de type définition

    def extract_from_text(self, text: str) -> list[GlossaryEntry]:
        """Extrait les définitions depuis un texte brut."""
        entries: list[GlossaryEntry] = []

        for pattern in self.TERM_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                entry = self._parse_match(match)
                if entry:
                    entries.append(entry)

        return entries

    def _extract_from_section(self, section: ParsedSection) -> list[GlossaryEntry]:
        """Extrait les entrées d'une section de glossaire."""
        entries: list[GlossaryEntry] = []
        text = section.content

        # Essayer chaque pattern
        for pattern in self.TERM_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                entry = self._parse_match(match)
                if entry:
                    entries.append(entry)

        # Extraction par lignes (fallback)
        if not entries:
            entries = self._extract_line_by_line(text)

        return entries

    def _parse_match(self, match: tuple) -> GlossaryEntry | None:
        """Parse un match regex en GlossaryEntry."""
        try:
            if len(match) == 3:
                term_fr, term_en, definition = match
                term_fr = term_fr.strip()
                term_en = term_en.strip()
                definition = definition.strip()
            elif len(match) == 2:
                term_fr, definition = match
                term_fr = term_fr.strip()
                term_en = None
                definition = definition.strip()
            else:
                return None

            # Validation minimale
            if not term_fr or not definition:
                return None
            if len(term_fr) < 2 or len(definition) < 10:
                return None

            # Extraire la référence normative
            standard_ref = self._extract_standard_ref(definition)

            return GlossaryEntry(
                term_fr=term_fr,
                term_en=term_en,
                definition_fr=definition,
                standard_ref=standard_ref,
            )
        except Exception:
            return None

    def _extract_line_by_line(self, text: str) -> list[GlossaryEntry]:
        """Extraction fallback ligne par ligne."""
        entries: list[GlossaryEntry] = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Pattern simple: Terme : définition
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    term = parts[0].strip().strip("-•* ")
                    definition = parts[1].strip()

                    if len(term) >= 2 and len(definition) >= 10:
                        # Chercher un terme anglais entre parenthèses
                        term_en_match = re.search(r"\(([^)]+)\)", term)
                        term_en = None
                        if term_en_match:
                            term_en = term_en_match.group(1)
                            term = re.sub(r"\s*\([^)]+\)", "", term).strip()

                        entries.append(
                            GlossaryEntry(
                                term_fr=term,
                                term_en=term_en,
                                definition_fr=definition,
                                standard_ref=self._extract_standard_ref(definition),
                            )
                        )

        return entries

    def _extract_standard_ref(self, text: str) -> str | None:
        """Extrait la référence normative du texte."""
        match = self.STANDARD_REF_PATTERN.search(text)
        if match:
            ref = match.group(1)
            # Normaliser
            ref = ref.replace(" ", "/").replace("//", "/")
            return ref
        return None

    def _deduplicate(self, entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
        """Déduplique les entrées par terme FR."""
        seen: dict[str, GlossaryEntry] = {}

        for entry in entries:
            key = entry.term_fr.lower()
            if key not in seen:
                seen[key] = entry
            else:
                # Garder l'entrée la plus complète
                existing = seen[key]
                if (
                    len(entry.definition_fr) > len(existing.definition_fr)
                    or entry.term_en
                    and not existing.term_en
                    or entry.standard_ref
                    and not existing.standard_ref
                ):
                    seen[key] = entry

        return list(seen.values())

    def validate_entry(self, entry: GlossaryEntry) -> bool:
        """Valide qu'une entrée est pertinente pour le GRI.

        Une entrée est valide si:
        1. C'est un terme technique connu (KNOWN_TERMS)
        2. OU elle a une référence standard (ISO, IEEE, etc.)
        3. OU le terme est court (< 50 chars) ET la définition est substantielle (> 50 chars)
        """
        term_lower = entry.term_fr.lower()

        # Critère 1: Terme connu dans le domaine
        for known in self.KNOWN_TERMS:
            if known in term_lower or term_lower in known:
                return True

        # Critère 2: A une référence standard
        if entry.standard_ref:
            return True

        # Critère 3: Terme court avec définition substantielle
        # (évite les titres de sections qui sont longs)
        if len(entry.term_fr) < 50 and len(entry.definition_fr) > 50:
            # Vérifier que ça ressemble à une définition (contient des mots clés)
            def_lower = entry.definition_fr.lower()
            definition_markers = [
                "est ",
                "sont ",
                "désigne",
                "représente",
                "permet",
                "consiste",
                "défini",
                "process",
                "système",
                "méthode",
            ]
            if any(marker in def_lower for marker in definition_markers):
                return True

        return False

    def save_to_json(
        self,
        entries: list[GlossaryEntry],
        output_path: str | Path,
    ) -> None:
        """Sauvegarde le glossaire en JSON."""
        import json

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [entry.model_dump() for entry in entries]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log.info(
            "gri.glossary.saved",
            path=str(path),
            entries=len(entries),
        )

    def load_from_json(self, input_path: str | Path) -> list[GlossaryEntry]:
        """Charge le glossaire depuis un JSON."""
        import json

        path = Path(input_path)
        if not path.exists():
            return []

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        entries = [GlossaryEntry(**item) for item in data]

        log.info(
            "gri.glossary.loaded",
            path=str(path),
            entries=len(entries),
        )

        return entries
