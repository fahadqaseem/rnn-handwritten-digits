"""
make_pdf.py — Generate "Assignment 4 Submission_hr4625.pdf" from README.md.

Uses system Arial Unicode + Courier New fonts (both available on macOS)
for full UTF-8 support. No LaTeX or additional dependencies needed.
"""
from fpdf import FPDF, XPos, YPos
from PIL import Image
import re, os

# ── System font paths (macOS) ─────────────────────────────────────────────────
SYS_FONTS = "/System/Library/Fonts/Supplemental"
FONT_REGULAR     = f"{SYS_FONTS}/Arial Unicode.ttf"
FONT_BOLD        = f"{SYS_FONTS}/Arial Bold.ttf"
FONT_ITALIC      = f"{SYS_FONTS}/Arial Narrow Italic.ttf"
FONT_MONO        = f"{SYS_FONTS}/Courier New.ttf"
FONT_MONO_BOLD   = f"{SYS_FONTS}/Courier New Bold.ttf"

# ── constants ─────────────────────────────────────────────────────────────────
README    = "README.md"
OUT       = "Assignment 4 Submission_hr4625.pdf"
MARGIN    = 15
LINE_H    = 5.5
CODE_H    = 4.6
PAGE_W    = 210
CONTENT_W = PAGE_W - 2 * MARGIN

BLUE  = (30,  100, 200)
DARK  = (30,   30,  30)
MID   = (80,   80,  80)
LIGHT = (220, 220, 220)
CODE_BG=(245, 245, 245)


def clean(text: str) -> str:
    """Strip common Markdown inline markers, return plain Unicode string."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'`(.+?)`',       r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    return text


class PDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(MARGIN, 20, MARGIN)
        # Register Unicode-capable system fonts
        self.add_font("Body",    fname=FONT_REGULAR)
        self.add_font("Body",    style="B", fname=FONT_BOLD)
        self.add_font("Body",    style="I", fname=FONT_ITALIC)
        self.add_font("Mono",    fname=FONT_MONO)
        self.add_font("Mono",    style="B", fname=FONT_MONO_BOLD)

    def header(self):
        self.set_font("Body", "B", 8)
        self.set_text_color(*MID)
        self.cell(0, 6, "Homework 4: RNN Digit Classifier  |  hr4625  |  CSC 5991",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*LIGHT)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.set_font("Body", "", 8)
        self.set_text_color(*MID)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    # ── layout primitives ────────────────────────────────────────────────────
    def h1(self, text):
        self.ln(3)
        self.set_font("Body", "B", 18)
        self.set_text_color(*DARK)
        self.multi_cell(0, 9, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.8)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    def h2(self, text):
        self.ln(4)
        self.set_font("Body", "B", 13)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 7, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.3)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(2)

    def h3(self, text):
        self.ln(3)
        self.set_font("Body", "B", 11)
        self.set_text_color(*DARK)
        self.multi_cell(0, 6, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body(self, text):
        self.set_font("Body", "", 10)
        self.set_text_color(*DARK)
        self.multi_cell(CONTENT_W, LINE_H, clean(text),
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bullet(self, text, level=0):
        indent = 5 * (level + 1)
        self.set_font("Body", "", 10)
        self.set_text_color(*DARK)
        content = re.sub(r'^[\s\-\*\+]+', '', text).strip()
        self.set_x(MARGIN + indent)
        self.multi_cell(CONTENT_W - indent, LINE_H, f"• {clean(content)}",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def numbered(self, num, text):
        indent = 6
        self.set_font("Body", "", 10)
        self.set_text_color(*DARK)
        self.set_x(MARGIN + indent)
        self.multi_cell(CONTENT_W - indent, LINE_H, f"{num}. {clean(text)}",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def code_block(self, code_lines):
        self.ln(1)
        block_h = len(code_lines) * CODE_H + 4
        if self.get_y() + block_h > 268:
            self.add_page()
        self.set_fill_color(*CODE_BG)
        self.rect(MARGIN, self.get_y(), CONTENT_W, block_h, style="F")
        self.set_font("Mono", "", 8)
        self.set_text_color(*DARK)
        y0 = self.get_y() + 2
        for line in code_lines:
            self.set_xy(MARGIN + 2, y0)
            display = line[:105] + ("…" if len(line) > 105 else "")
            self.cell(CONTENT_W - 4, CODE_H, display)
            y0 += CODE_H
        self.set_y(y0 + 2)
        self.ln(1)

    def table(self, headers, rows):
        if not headers:
            return
        self.ln(2)
        n     = len(headers)
        col_w = CONTENT_W / n

        # header row
        self.set_fill_color(*BLUE)
        self.set_text_color(255, 255, 255)
        self.set_font("Body", "B", 8.5)
        for h in headers:
            self.cell(col_w, 6.5, clean(h).strip(),
                      border=0, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln(6.5)

        # data rows
        self.set_font("Body", "", 8.5)
        for ri, row in enumerate(rows):
            fill = (243, 247, 255) if ri % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*fill)
            self.set_text_color(*DARK)
            for cell in row:
                self.cell(col_w, 5.5, clean(str(cell)).strip(),
                          border=0, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.ln(5.5)

        self.set_draw_color(*LIGHT)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    def embed_image(self, path, caption=""):
        if not os.path.exists(path):
            self.body(f"[Image not found: {path}]")
            return
        try:
            with Image.open(path) as im:
                iw, ih = im.size
            aspect    = ih / iw
            display_w = min(CONTENT_W, 155)
            display_h = display_w * aspect
            if self.get_y() + display_h > 268:
                self.add_page()
            x = MARGIN + (CONTENT_W - display_w) / 2
            self.image(path, x=x, w=display_w)
            if caption:
                self.set_font("Body", "I", 8.5)
                self.set_text_color(*MID)
                self.multi_cell(0, 5, caption, align="C",
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(3)
        except Exception as e:
            self.body(f"[Could not load image: {path}]")

    def horizontal_rule(self):
        self.ln(2)
        self.set_draw_color(*LIGHT)
        self.set_line_width(0.3)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    def blockquote(self, text):
        self.set_font("Body", "I", 9.5)
        self.set_text_color(*MID)
        y_start = self.get_y()
        self.set_x(MARGIN + 5)
        self.multi_cell(CONTENT_W - 5, LINE_H, clean(text),
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        y_end = self.get_y()
        self.set_draw_color(*BLUE)
        self.set_line_width(1.2)
        self.line(MARGIN, y_start, MARGIN, y_end)
        self.ln(1)


def parse_table(lines):
    result = []
    for line in lines:
        if re.match(r'^\|[\-| :]+\|$', line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        result.append(cells)
    if not result:
        return [], []
    return result[0], result[1:]


def render(pdf: PDF, lines):
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')

        if re.match(r'^# [^#]', line):
            pdf.h1(line[2:]); i += 1; continue

        if re.match(r'^## [^#]', line):
            pdf.h2(line[3:]); i += 1; continue

        if re.match(r'^### ', line):
            pdf.h3(line[4:]); i += 1; continue

        if re.match(r'^-{3,}$', line.strip()):
            pdf.horizontal_rule(); i += 1; continue

        if line.strip().startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i].rstrip('\n'))
                i += 1
            pdf.code_block(code_lines)
            i += 1; continue

        if line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].startswith('|'):
                table_lines.append(lines[i].rstrip('\n'))
                i += 1
            headers, rows = parse_table(table_lines)
            pdf.table(headers, rows)
            continue

        m = re.match(r'^!\[(.+?)\]\((.+?)\)$', line.strip())
        if m:
            pdf.embed_image(m.group(2), m.group(1)); i += 1; continue

        if line.startswith('> '):
            bq = line[2:]
            i += 1
            while i < len(lines) and lines[i].startswith('> '):
                bq += ' ' + lines[i][2:].strip(); i += 1
            pdf.blockquote(re.sub(r'\*(.+?)\*', r'\1', bq))
            continue

        if re.match(r'^(\s*)([-*+])\s', line):
            level   = (len(line) - len(line.lstrip())) // 2
            content = re.sub(r'^(\s*[-*+])\s+', '', line)
            pdf.bullet(content, level=level); i += 1; continue

        m = re.match(r'^(\d+)\.\s+(.+)', line)
        if m:
            pdf.numbered(m.group(1), m.group(2)); i += 1; continue

        if line.strip() == '':
            pdf.ln(2); i += 1; continue

        pdf.body(line); i += 1


# ── Build PDF ────────────────────────────────────────────────────────────────
pdf = PDF()
pdf.add_page()

with open(README, encoding="utf-8") as f:
    lines = f.readlines()

render(pdf, lines)
pdf.output(OUT)
print(f"PDF written: {OUT}  ({os.path.getsize(OUT):,} bytes)")

