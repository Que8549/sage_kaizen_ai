"""
tests/test_document_parser.py

Unit tests for document_parser.py.

Covers extension routing, the UTF-8 -> latin-1 decode fallback, the two
character caps (per-document and total-across-turn), and the <document> block
rendering that chat_service prepends to the user message.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from document_parser import (
    ACCEPTED_EXTENSIONS,
    PER_DOCUMENT_CHAR_LIMIT,
    TOTAL_CHAR_LIMIT,
    DocumentAttachment,
    _extract_docx,
    _extract_plain_text,
    _extract_xlsx,
    _file_extension,
    format_document_context,
    is_supported,
    parse_document,
)


# ---------------------------------------------------------------------------
# Extension helpers
# ---------------------------------------------------------------------------

class TestFileExtension:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("main.py", "py"),
            ("README.MD", "md"),
            ("archive.tar.gz", "gz"),
            ("Dockerfile", ""),
            ("", ""),
            (".gitignore", "gitignore"),
            ("weird.PS1", "ps1"),
        ],
    )
    def test_extension_parsing(self, filename, expected):
        assert _file_extension(filename) == expected


class TestIsSupported:
    @pytest.mark.parametrize(
        "filename", ["a.py", "b.docx", "c.xlsx", "d.sql", "e.yaml", "f.ts"]
    )
    def test_supported(self, filename):
        assert is_supported(filename) is True

    @pytest.mark.parametrize(
        "filename", ["a.exe", "b.pdf", "c.png", "d.mp3", "Dockerfile", "e.zip"]
    )
    def test_unsupported(self, filename):
        assert is_supported(filename) is False

    def test_accepted_extensions_has_no_leading_dots(self):
        assert all(not e.startswith(".") for e in ACCEPTED_EXTENSIONS)

    def test_office_types_are_accepted(self):
        assert {"docx", "xlsx"} <= ACCEPTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Plain-text extraction
# ---------------------------------------------------------------------------

class TestExtractPlainText:
    def test_decodes_utf8(self):
        assert _extract_plain_text("héllo — wörld".encode("utf-8"), "a.txt") == "héllo — wörld"

    def test_falls_back_to_latin1_on_invalid_utf8(self):
        # 0xFF is never valid as a UTF-8 lead byte; latin-1 maps it to U+00FF.
        result = _extract_plain_text(b"caf\xff", "a.txt")
        assert result == "cafÿ"

    def test_latin1_fallback_is_lossless_for_all_bytes(self):
        raw = bytes(range(256))
        # Must not raise, and must preserve length (latin-1 is byte-for-byte).
        assert len(_extract_plain_text(raw, "a.bin")) == 256

    def test_empty_bytes(self):
        assert _extract_plain_text(b"", "a.txt") == ""


# ---------------------------------------------------------------------------
# parse_document — routing, typing, truncation
# ---------------------------------------------------------------------------

class TestParseDocument:
    def test_returns_none_for_unsupported(self):
        assert parse_document(b"data", "image.png") is None

    def test_plain_text_roundtrip(self):
        att = parse_document(b"print('hi')", "main.py")
        assert att is not None
        assert att.filename == "main.py"
        assert att.doc_type == "python"
        assert att.content == "print('hi')"
        assert att.char_count == len("print('hi')")
        assert att.truncated is False

    @pytest.mark.parametrize(
        "filename,doc_type",
        [
            ("a.md", "markdown"),
            ("a.sql", "sql"),
            ("a.ps1", "powershell"),
            ("a.cs", "csharp"),
            ("a.yml", "yaml"),
            ("a.htm", "html"),
            ("a.rs", "rust"),
        ],
    )
    def test_doc_type_labels(self, filename, doc_type):
        att = parse_document(b"x", filename)
        assert att is not None and att.doc_type == doc_type

    def test_truncates_at_per_document_limit(self):
        raw = (b"a" * (PER_DOCUMENT_CHAR_LIMIT + 5_000))
        att = parse_document(raw, "big.txt")
        assert att is not None
        assert att.truncated is True
        assert att.content.startswith("a" * 100)
        assert "truncated" in att.content
        # char_count reflects the post-truncation content, per the docstring.
        assert att.char_count == len(att.content)
        assert att.char_count > PER_DOCUMENT_CHAR_LIMIT  # limit + notice

    def test_exactly_at_limit_is_not_truncated(self):
        att = parse_document(b"a" * PER_DOCUMENT_CHAR_LIMIT, "edge.txt")
        assert att is not None
        assert att.truncated is False
        assert att.char_count == PER_DOCUMENT_CHAR_LIMIT


# ---------------------------------------------------------------------------
# Office extractors — exercised through mocks (no binary fixtures needed)
# ---------------------------------------------------------------------------

class TestExtractDocx:
    def test_extracts_paragraphs_and_tables(self):
        para = MagicMock(text="  Heading  ")
        empty = MagicMock(text="   ")
        cell_a, cell_b = MagicMock(text="A"), MagicMock(text="B")
        row = MagicMock(cells=[cell_a, cell_b])
        table = MagicMock(rows=[row])
        doc = MagicMock(paragraphs=[para, empty], tables=[table])

        fake_docx = MagicMock()
        fake_docx.Document.return_value = doc
        with patch.dict("sys.modules", {"docx": fake_docx}):
            out = _extract_docx(b"fake", "a.docx")

        assert "Heading" in out
        assert "A | B" in out
        assert "   " not in out.split("\n")  # empty paragraph dropped

    def test_deduplicates_merged_cells(self):
        # python-docx repeats cell text across a merged span.
        cells = [MagicMock(text="M"), MagicMock(text="M"), MagicMock(text="N")]
        doc = MagicMock(paragraphs=[], tables=[MagicMock(rows=[MagicMock(cells=cells)])])
        fake_docx = MagicMock()
        fake_docx.Document.return_value = doc
        with patch.dict("sys.modules", {"docx": fake_docx}):
            out = _extract_docx(b"fake", "a.docx")
        assert "M | N" in out

    def test_returns_error_marker_on_failure(self):
        fake_docx = MagicMock()
        fake_docx.Document.side_effect = ValueError("corrupt zip")
        with patch.dict("sys.modules", {"docx": fake_docx}):
            out = _extract_docx(b"garbage", "a.docx")
        assert out.startswith("[Error extracting document content:")
        assert "corrupt zip" in out

    def test_parse_document_routes_docx(self):
        with patch("document_parser._extract_docx", return_value="text") as m:
            att = parse_document(b"x", "report.docx")
        m.assert_called_once()
        assert att is not None and att.doc_type == "word-document"


class TestExtractXlsx:
    def _wb(self, sheets: dict[str, list[tuple]]):
        wb = MagicMock()
        wb.sheetnames = list(sheets)
        wb.__getitem__.side_effect = lambda name: MagicMock(
            iter_rows=MagicMock(return_value=iter(sheets[name]))
        )
        return wb

    def test_renders_sheets_with_headers(self):
        wb = self._wb({"Data": [("a", 1), ("b", 2)]})
        fake = MagicMock()
        fake.load_workbook.return_value = wb
        with patch.dict("sys.modules", {"openpyxl": fake}):
            out = _extract_xlsx(b"fake", "a.xlsx")
        assert "### Sheet: Data" in out
        assert "a | 1" in out and "b | 2" in out

    def test_skips_fully_empty_rows(self):
        wb = self._wb({"S": [(None, None), ("", "  "), ("x", None)]})
        fake = MagicMock()
        fake.load_workbook.return_value = wb
        with patch.dict("sys.modules", {"openpyxl": fake}):
            out = _extract_xlsx(b"fake", "a.xlsx")
        assert out.count("\n") == 1  # header + the single non-empty row
        assert "x | " in out

    def test_sheet_with_no_rows_emits_no_header(self):
        wb = self._wb({"Empty": []})
        fake = MagicMock()
        fake.load_workbook.return_value = wb
        with patch.dict("sys.modules", {"openpyxl": fake}):
            out = _extract_xlsx(b"fake", "a.xlsx")
        assert out == ""

    def test_returns_error_marker_on_failure(self):
        fake = MagicMock()
        fake.load_workbook.side_effect = OSError("bad file")
        with patch.dict("sys.modules", {"openpyxl": fake}):
            out = _extract_xlsx(b"x", "a.xlsx")
        assert out.startswith("[Error extracting workbook content:")

    def test_parse_document_routes_xlsx(self):
        with patch("document_parser._extract_xlsx", return_value="cells") as m:
            att = parse_document(b"x", "book.xlsx")
        m.assert_called_once()
        assert att is not None and att.doc_type == "excel-workbook"


# ---------------------------------------------------------------------------
# format_document_context — the block chat_service injects
# ---------------------------------------------------------------------------

def _att(name: str, content: str, doc_type: str = "text") -> DocumentAttachment:
    return DocumentAttachment(
        filename=name, content=content, doc_type=doc_type,
        char_count=len(content), truncated=False,
    )


class TestFormatDocumentContext:
    def test_empty_returns_empty_string(self):
        assert format_document_context(()) == ""

    def test_single_document_block_shape(self):
        out = format_document_context((_att("a.py", "code", "python"),))
        assert out.startswith('<document filename="a.py" type="python" chars="4">')
        assert out.endswith("</document>")
        assert "\ncode\n" in out

    def test_multiple_documents_separated(self):
        out = format_document_context((_att("a.txt", "one"), _att("b.txt", "two")))
        assert out.count("<document ") == 2
        assert "</document>\n\n<document" in out

    def test_total_budget_truncates_later_document(self):
        first = _att("big.txt", "a" * (TOTAL_CHAR_LIMIT - 10))
        second = _att("next.txt", "b" * 500)
        out = format_document_context((first, second))
        assert "truncated" in out
        # Second document keeps only the 10 chars of remaining budget.
        assert out.count("b") <= 20

    def test_document_past_exhausted_budget_is_omitted(self):
        first = _att("big.txt", "a" * TOTAL_CHAR_LIMIT)
        second = _att("dropped.txt", "b" * 100)
        out = format_document_context((first, second))
        assert "omitted — total document budget exhausted" in out
        assert '<document filename="dropped.txt"' not in out

    def test_chars_attribute_reports_original_count_not_budget_trimmed(self):
        """`chars` is the attachment's own count; budget trimming is separate."""
        att = _att("a.txt", "x" * 100)
        out = format_document_context((att,))
        assert 'chars="100"' in out
