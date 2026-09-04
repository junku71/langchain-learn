from __future__ import annotations

from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle


def korean_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/malgun.ttf")
    bold = Path("C:/Windows/Fonts/malgunbd.ttf")
    if not regular.exists():
        raise RuntimeError("한글 PDF 생성에 필요한 맑은 고딕 글꼴을 찾을 수 없습니다.")
    if "Malgun" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Malgun", str(regular)))
    if "MalgunBold" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("MalgunBold", str(bold if bold.exists() else regular)))
    return "Malgun", "MalgunBold"


def styles():
    regular, bold = korean_fonts()
    sheet = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("KoreanTitle", parent=sheet["Title"], fontName=bold, fontSize=20, leading=26, textColor=colors.HexColor("#173d78"), alignment=TA_CENTER, spaceAfter=14),
        "heading": ParagraphStyle("KoreanHeading", parent=sheet["Heading2"], fontName=bold, fontSize=13, leading=18, textColor=colors.HexColor("#173d78"), spaceBefore=12, spaceAfter=7),
        "body": ParagraphStyle("KoreanBody", parent=sheet["BodyText"], fontName=regular, fontSize=8.5, leading=12),
        "small": ParagraphStyle("KoreanSmall", parent=sheet["BodyText"], fontName=regular, fontSize=6.5, leading=9),
    }


def para(value, style) -> Paragraph:
    text = str(value if value not in (None, "") else "-")
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def report_table(rows, widths, *, header=True, font_size=7) -> Table:
    regular, bold = korean_fonts()
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), regular), ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c5d6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend([("FONTNAME", (0, 0), (-1, 0), bold), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0fa"))])
    table.setStyle(TableStyle(commands))
    return table


def pdf_document(path: Path, *, landscape_page=False) -> SimpleDocTemplate:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SimpleDocTemplate(str(path), pagesize=landscape(A4) if landscape_page else A4, leftMargin=28, rightMargin=28, topMargin=28, bottomMargin=28, title=path.stem)
