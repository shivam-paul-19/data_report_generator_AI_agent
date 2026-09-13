import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
import io

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas
from PIL import Image as PILImage


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print total page numbers
    along with a professional running header and footer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Executive Data Intelligence & Analytical Report")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)

        # Running Footer (all pages)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "AI-assisted analysis & report generation — results should be independently verified.")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)

        self.restoreState()


class PDFReportMaker:
    """
    Converts full Markdown data reports and embedded Matplotlib/Seaborn charts
    into publication-quality PDF documents using ReportLab.
    """

    def __init__(self, page_size=A4, margin: float = 54):
        self.page_size = page_size
        self.margin = margin
        self.usable_width = self.page_size[0] - (2 * self.margin)
        self.styles = self._create_styles()

    def _create_styles(self) -> dict:
        base_styles = getSampleStyleSheet()

        styles = {
            "Title": ParagraphStyle(
                "CustomTitle",
                parent=base_styles["Title"],
                fontName="Helvetica-Bold",
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#0f172a"),
                alignment=0,
                spaceAfter=12,
            ),
            "Heading1": ParagraphStyle(
                "CustomH1",
                parent=base_styles["Heading1"],
                fontName="Helvetica-Bold",
                fontSize=14,
                leading=18,
                textColor=colors.HexColor("#1e3a8a"),
                spaceBefore=14,
                spaceAfter=8,
                keepWithNext=True,
            ),
            "Heading2": ParagraphStyle(
                "CustomH2",
                parent=base_styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=12,
                leading=16,
                textColor=colors.HexColor("#1e40af"),
                spaceBefore=12,
                spaceAfter=6,
                keepWithNext=True,
            ),
            "Heading3": ParagraphStyle(
                "CustomH3",
                parent=base_styles["Heading3"],
                fontName="Helvetica-Bold",
                fontSize=10.5,
                leading=14,
                textColor=colors.HexColor("#334155"),
                spaceBefore=8,
                spaceAfter=4,
                keepWithNext=True,
            ),
            "Body": ParagraphStyle(
                "CustomBody",
                parent=base_styles["BodyText"],
                fontName="Helvetica",
                fontSize=9,
                leading=13.5,
                textColor=colors.HexColor("#1e293b"),
                spaceAfter=6,
            ),
            "Bullet": ParagraphStyle(
                "CustomBullet",
                parent=base_styles["BodyText"],
                fontName="Helvetica",
                fontSize=9,
                leading=13,
                textColor=colors.HexColor("#1e293b"),
                leftIndent=15,
                firstLineIndent=-10,
                spaceAfter=3,
            ),
            "Callout": ParagraphStyle(
                "CustomCallout",
                parent=base_styles["BodyText"],
                fontName="Helvetica-Oblique",
                fontSize=8.5,
                leading=12.5,
                textColor=colors.HexColor("#475569"),
                leftIndent=14,
                spaceBefore=4,
                spaceAfter=6,
            ),
            "TableCell": ParagraphStyle(
                "CustomTableCell",
                parent=base_styles["Normal"],
                fontName="Helvetica",
                fontSize=8,
                leading=10.5,
                textColor=colors.HexColor("#1e293b"),
            ),
            "TableHeader": ParagraphStyle(
                "CustomTableHeader",
                parent=base_styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10.5,
                textColor=colors.white,
            ),
            "Caption": ParagraphStyle(
                "CustomCaption",
                parent=base_styles["Normal"],
                fontName="Helvetica-Oblique",
                fontSize=8,
                leading=10.5,
                textColor=colors.HexColor("#64748b"),
                alignment=1,  # Centered
                spaceBefore=4,
                spaceAfter=8,
            ),
        }
        return styles

    @staticmethod
    def _clean_markdown_formatting(text: str) -> str:
        """
        Translates common markdown syntax to ReportLab XML tags:
        **bold** -> <b>bold</b>
        *italic* -> <i>italic</i>
        `code`   -> <font face="Courier">code</font>
        """
        # Escape XML entities first, except already formatted ones
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Convert back our markdown syntax
        # Bold
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

        # Italic
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        text = re.sub(r"_(.+?)_", r"<i>\1</i>", text)

        # Inline code
        text = re.sub(r"`(.+?)`", r'<font face="Courier" color="#b91c1c">\1</font>', text)

        return text

    def _parse_table(self, table_lines: List[str]) -> Optional[Table]:
        """
        Parse Markdown table lines into a styled ReportLab Table flowable.
        """
        rows_data = []
        for line in table_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("|---") or re.match(r"^\|(\s*:?-+:?\s*\|)+$", stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells:
                rows_data.append(cells)

        if not rows_data:
            return None

        num_cols = max(len(r) for r in rows_data)
        # Normalize column counts
        for r in rows_data:
            while len(r) < num_cols:
                r.append("")

        # Convert cells to Paragraphs for auto-wrapping
        table_paragraphs = []
        for row_idx, row in enumerate(rows_data):
            p_row = []
            is_header = (row_idx == 0)
            style_to_use = self.styles["TableHeader"] if is_header else self.styles["TableCell"]
            for cell_text in row:
                formatted_text = self._clean_markdown_formatting(cell_text)
                p_row.append(Paragraph(formatted_text, style_to_use))
            table_paragraphs.append(p_row)

        col_width = self.usable_width / max(num_cols, 1)
        col_widths = [col_width] * num_cols

        t = Table(table_paragraphs, colWidths=col_widths, repeatRows=1)
        t_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]

        # Alternating row background
        for i in range(1, len(rows_data)):
            if i % 2 == 0:
                t_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafc")))

        t.setStyle(TableStyle(t_style))
        return t

    def _prepare_image_flowable(self, image_input: Union[str, Path, bytes, io.BytesIO], caption_text: str = "") -> List:
        """
        Creates an aspect-ratio-scaled ReportLab Image flowable and caption from a file path or in-memory bytes.
        """
        elements = []
        img_buffer = None

        try:
            if isinstance(image_input, bytes):
                img_buffer = io.BytesIO(image_input)
                pil_img = PILImage.open(img_buffer)
            elif isinstance(image_input, io.BytesIO):
                img_buffer = image_input
                img_buffer.seek(0)
                pil_img = PILImage.open(img_buffer)
            else:
                path = Path(image_input)
                if not path.exists():
                    return elements
                pil_img = PILImage.open(path)

            orig_w, orig_h = pil_img.size

            max_w = self.usable_width
            max_h = 240  # Max height in points

            # Scaling while preserving aspect ratio
            scale = min(max_w / orig_w, max_h / orig_h)
            display_w = orig_w * scale
            display_h = orig_h * scale

            if img_buffer is not None:
                img_buffer.seek(0)
                rl_img = RLImage(img_buffer, width=display_w, height=display_h)
            else:
                rl_img = RLImage(str(image_input), width=display_w, height=display_h)

            rl_img.hAlign = "CENTER"
            elements.append(Spacer(1, 6))
            elements.append(rl_img)
            if caption_text:
                elements.append(Paragraph(f"Figure: {caption_text}", self.styles["Caption"]))
            else:
                elements.append(Spacer(1, 6))
        except Exception as e:
            logger.error(f"Error loading image for PDF: {e}")

        return elements

    def convert_markdown_to_pdf(
        self,
        markdown_text: str,
        output_pdf_path: Optional[str] = None,
        charts: Optional[List[dict]] = None
    ) -> Union[str, bytes]:
        """
        Parses full markdown report text and embedded charts, generating a complete PDF.
        Returns the output path or in-memory PDF bytes.
        """
        story = []

        # Track charts that might need to be embedded if not already in markdown
        chart_paths_seen = set()

        lines = markdown_text.split("\n")
        in_table = False
        table_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check for Table lines
            if stripped.startswith("|") and stripped.endswith("|"):
                in_table = True
                table_lines.append(stripped)
                i += 1
                continue
            else:
                if in_table:
                    # Flush table
                    tbl = self._parse_table(table_lines)
                    if tbl:
                        story.append(tbl)
                        story.append(Spacer(1, 6))
                    table_lines = []
                    in_table = False

            if not stripped:
                story.append(Spacer(1, 3))
                i += 1
                continue

            # Horizontal rule
            if stripped in ["---", "***", "___"]:
                story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceBefore=6, spaceAfter=8))
                i += 1
                continue

            # Markdown Image: ![caption](path)
            img_match = re.match(r"^!\[(.*?)\]\((.*?)\)", stripped)
            if img_match:
                alt_text = img_match.group(1)
                img_path = img_match.group(2).strip()
                chart_paths_seen.add(img_path)
                story.extend(self._prepare_image_flowable(img_path, caption_text=alt_text))
                i += 1
                continue

            # Headers
            if stripped.startswith("# "):
                title_text = self._clean_markdown_formatting(stripped[2:].strip())
                story.append(Paragraph(title_text, self.styles["Title"]))
                story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceBefore=2, spaceAfter=10))
            elif stripped.startswith("## "):
                h1_text = self._clean_markdown_formatting(stripped[3:].strip())
                story.append(Paragraph(h1_text, self.styles["Heading1"]))
            elif stripped.startswith("### "):
                h2_text = self._clean_markdown_formatting(stripped[4:].strip())
                story.append(Paragraph(h2_text, self.styles["Heading2"]))
            elif stripped.startswith("#### "):
                h3_text = self._clean_markdown_formatting(stripped[5:].strip())
                story.append(Paragraph(h3_text, self.styles["Heading3"]))
            elif stripped.startswith("> "):
                callout_text = self._clean_markdown_formatting(stripped[2:].strip())
                story.append(Paragraph(callout_text, self.styles["Callout"]))
            elif stripped.startswith(("- ", "* ", "+ ")):
                bullet_text = "&bull; " + self._clean_markdown_formatting(stripped[2:].strip())
                story.append(Paragraph(bullet_text, self.styles["Bullet"]))
            elif re.match(r"^\d+\.\s+", stripped):
                num_prefix = re.match(r"^\d+\.\s+", stripped).group(0)
                list_text = f"<b>{num_prefix}</b>" + self._clean_markdown_formatting(stripped[len(num_prefix):].strip())
                story.append(Paragraph(list_text, self.styles["Bullet"]))
            else:
                body_text = self._clean_markdown_formatting(stripped)
                story.append(Paragraph(body_text, self.styles["Body"]))

            i += 1

        # Flush any trailing table
        if in_table and table_lines:
            tbl = self._parse_table(table_lines)
            if tbl:
                story.append(tbl)
                story.append(Spacer(1, 6))

        # If there are additional charts not explicitly placed in markdown, append in an appendix
        if charts:
            unplaced = [c for c in charts if c.get("path") and str(c["path"]) not in chart_paths_seen]
            if unplaced:
                story.append(Spacer(1, 10))
                story.append(Paragraph("📊 Visual Charts & Analytics Appendix", self.styles["Heading1"]))
                story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1e3a8a"), spaceBefore=2, spaceAfter=8))
                for c in unplaced:
                    story.append(Paragraph(f"<b>{c.get('title', 'Analytical Chart')}</b>", self.styles["Heading2"]))
                    story.extend(self._prepare_image_flowable(c["path"], caption_text=c.get("caption", "")))

        # Build Document
        if output_pdf_path:
            doc = SimpleDocTemplate(
                output_pdf_path,
                pagesize=self.page_size,
                leftMargin=self.margin,
                rightMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin,
            )
            doc.build(story, canvasmaker=NumberedCanvas)
            return output_pdf_path
        else:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=self.page_size,
                leftMargin=self.margin,
                rightMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin,
            )
            doc.build(story, canvasmaker=NumberedCanvas)
            buffer.seek(0)
            return buffer.getvalue()


def make_pdf_from_report(
    markdown_report: str,
    output_path: Optional[str] = "executive_data_report.pdf",
    charts: Optional[List[dict]] = None
) -> Union[str, bytes]:
    """
    Convenience function to convert a markdown report and charts to a PDF.
    """
    maker = PDFReportMaker()
    return maker.convert_markdown_to_pdf(markdown_report, output_pdf_path=output_path, charts=charts)


if __name__ == "__main__":
    sample_md = """# 📊 Comprehensive Executive Data Intelligence & Analytical Report

## 1. 📋 Executive Summary, Strategic Context & Architecture
This analytical assessment provides an end-to-end evaluation of the enterprise dataset.

| Metric | Value | Status |
|---|---|---|
| Total Observations | 1,309 | Verified |
| Feature Dimensions | 14 | Optimal |
| Missing Rate | 4.2% | Low Risk |

## 2. 🔍 Data Health & Distributions
Below is the structural distribution and missingness audit.

- Missing values in `Age` require median imputation conditioned on passenger class.
- Multi-collinearity is minimal across primary indicators.
"""
    output_file = "test_output.pdf"
    make_pdf_from_report(sample_md, output_path=output_file)
    print(f"Generated test PDF at: {output_file}")
