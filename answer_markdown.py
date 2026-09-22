"""Normalize model math delimiters for Streamlit without changing calculations."""
import re


def normalize_math_markdown(text):
    # Leave code examples untouched. Handle math only in ordinary Markdown.
    parts = re.split(r'(```[\s\S]*?```|`[^`\n]*`)', text)
    for index in range(0, len(parts), 2):
        value = parts[index]
        value = re.sub(r'\\\((.*?)\\\)', lambda m: '$' + m[1].strip() + '$', value, flags=re.S)
        value = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: '$$' + m[1].strip() + '$$', value)

        def display(match):
            body = match[1].strip()
            # Markdown headings/citations must never become part of KaTeX.
            boundary = re.search(r'(?<!\\)#{1,6}\s|\n\s*[-*]\s', body)
            if boundary:
                equation, prose = body[:boundary.start()].strip(), body[boundary.start():].strip()
            else:
                equation, prose = body, ''
            citations = re.findall(r'\[(?:source[^\]]*|PDF p[^\]]*)\]', equation, re.I)
            equation = re.sub(r'\[(?:source[^\]]*|PDF p[^\]]*)\]', '', equation, flags=re.I).strip()
            result = '\n\n$$\n' + equation + '\n$$\n\n' if equation else '\n\n'
            return result + (' '.join(citations) + '\n\n' if citations else '') + (prose + '\n\n' if prose else '')

        value = re.sub(r'(?<![\\$])\$\$(?!\$)([\s\S]*?)(?<![\\$])\$\$(?!\$)[ \t]*', display, value)
        parts[index] = re.sub(r'\n{3,}', '\n\n', value)
    return ''.join(parts).strip()
