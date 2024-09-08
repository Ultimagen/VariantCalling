import re

import nbformat
from nbconvert.preprocessors import Preprocessor


class TocPreprocessor(Preprocessor):
    def preprocess(self, nb, resources):
        self.toc = []
        self.header_counters = [0] * 6  # We support up to 6 levels of headers
        nb.cells = [self.preprocess_cell(cell, resources, index) for index, cell in enumerate(nb.cells)]

        # Add custom text before the TOC
        custom_text = ""  # "# Report Introduction\n\nThis report contains an analysis of the ML model training.\n\n"
        custom_css = """
<style>
.toc-ol { counter-reset: item }
.toc-ol > li { display: block }
.toc-ol > li:before { content: counters(item, ".") ". "; counter-increment: item }
</style>
"""
        toc_html = self.generate_toc_html()
        toc_content = custom_css + custom_text + toc_html

        # Insert TOC at the placeholder or at the beginning
        self.insert_toc(nb, toc_content)

        return nb, resources

    def preprocess_cell(self, cell, resources, index):  # pylint: disable=unused-argument
        if cell.cell_type == "markdown":
            self.add_anchor_and_number_to_headings(cell)
        return cell

    def add_anchor_and_number_to_headings(self, cell):
        lines = cell.source.split("\n")
        new_lines = []
        inside_code_block = False

        for line in lines:
            # Check for start or end of a code block
            if line.strip().startswith("```"):
                inside_code_block = not inside_code_block

            # Only process headings outside of code blocks
            if not inside_code_block:
                match = re.match(r"^(#+)\s+(.*)", line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2)
                    self.update_header_counters(level)
                    section_number = self.get_section_number(level)
                    anchor = re.sub(r"\s+", "-", title.lower())
                    anchor = re.sub(r"[^\w\-]", "", anchor)  # Remove any characters that are not alphanumeric or dashes
                    numbered_title = f"{section_number}.  {title}"
                    line = f'<h{level} id="{anchor}">{numbered_title}</h{level}>'
                    self.toc.append((level, title, anchor))
            new_lines.append(line)

        cell.source = "\n".join(new_lines)

    def update_header_counters(self, level):
        self.header_counters[level - 1] += 1
        for i in range(level, len(self.header_counters)):
            self.header_counters[i] = 0

    def get_section_number(self, level):
        return ".".join(str(num) for num in self.header_counters[:level] if num > 0)

    def generate_toc_html(self):
        toc_html = '<div class="toc"><h2>Table of Contents</h2><ol class="toc-ol">'
        current_level = 1
        for level, title, anchor in self.toc:
            while current_level < level:
                toc_html += '<ol class="toc-ol">'
                current_level += 1
            while current_level > level:
                toc_html += "</ol>"
                current_level -= 1
            toc_html += f'<li><a href="#{anchor}">{title}</a></li>'
        while current_level > 1:
            toc_html += "</ol>"
            current_level -= 1
        toc_html += "</ol></div>"
        return toc_html

    def insert_toc(self, nb, toc_content):
        for cell in nb.cells:
            if cell.cell_type == "markdown" and "<!--TOC_PLACEHOLDER-->" in cell.source:
                cell.source = cell.source.replace("<!--TOC_PLACEHOLDER-->", toc_content)
                return

        # If placeholder is not found, insert TOC at the beginning
        toc_cell = nbformat.v4.new_markdown_cell(toc_content)
        nb.cells.insert(0, toc_cell)
