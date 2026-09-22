"""Presentation for the notebook. Illustrations are lightweight, native SVG."""
from html import escape
from pathlib import Path
import base64
import re

import streamlit as st


def _html(markup):
    # Streamlit sanitizes inline SVG. Image data URLs preserve the artwork and
    # its CSS animation without scripts, iframes, or a remote image service.
    def svg_image(match):
        svg = match.group(0)
        if 'xmlns=' not in svg:
            svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        svg = svg.replace('>', '''><style>
            svg { fill:none; stroke:#3f5646; stroke-width:1.3; stroke-linecap:round; stroke-linejoin:round; }
            text { stroke:none; }
            .filled { fill:#c07853; stroke:#c07853; }
            .electron-track { offset-path:path('M 315 155 a 135 49 0 1 0 -270 0 a 135 49 0 1 0 270 0'); offset-rotate:0deg; animation:orbit 12s linear infinite; }
            .electron-track.second { animation-duration:17s; animation-delay:-5s; }
            .electron-track.third { animation-duration:15s; animation-delay:-9s; }
            .nucleus-halo { transform-origin:180px 155px; animation:breathe 4s ease-in-out infinite; }
            @keyframes orbit { to { offset-distance:100%; } }
            @keyframes breathe { 0%,100% { transform:scale(.85); opacity:.08; } 50% { transform:scale(1.25); opacity:.13; } }
            @media(prefers-reduced-motion:reduce) { * { animation:none!important; } }
        </style>''', 1)
        if 'atom-system' in svg:
            svg = svg.replace('stroke:#3f5646;', 'stroke:none;', 1)
        encoded = base64.b64encode(svg.encode()).decode()
        if 'atom-system' in svg:
            # A separate still image also handles browsers that don't propagate
            # a changed motion preference into an already loaded SVG image.
            still = svg.replace('</style>', '* { animation:none!important; }</style>')
            static_encoded = base64.b64encode(still.encode()).decode()
            return (
                f'<img class="svg-motion" src="data:image/svg+xml;base64,{encoded}" alt="" aria-hidden="true" />'
                f'<img class="svg-still" src="data:image/svg+xml;base64,{static_encoded}" alt="" aria-hidden="true" />'
            )
        return f'<img src="data:image/svg+xml;base64,{encoded}" alt="" aria-hidden="true" />'
    st.html(re.sub(r'<svg\b.*?</svg>', svg_image, markup, flags=re.DOTALL))


def load_styles():
    st.html(Path(__file__).parent / "assets" / "notebook.css")


def brand():
    st.html('''<div class="brand"><span class="brand-mark">p<span>n</span><i></i></span>
        <div>physics notebook<span>A LITTLE CURIOSITY GOES A LONG WAY</span></div></div>''')


def library_card():
    st.html('''<div class="library-label">ON YOUR DESK <span>01</span></div>
    <div class="book-cover"><div class="book-edition">THE SCIENCE COLLECTION <span>IX</span></div>
    <div class="book-title">Physics<span>Understand the everyday.</span></div>
    <div class="book-orbits"><i></i><i></i><i></i><b></b></div>
    <div class="book-bottom">GRADE NINE <span>↗</span></div></div>
    <div class="library-caption"><b>Your physics textbook</b><span>200 pages · ready to explore</span></div>''')


def masthead():
    st.html('''<div class="masthead"><span>YOUR SPACE TO FIGURE THINGS OUT</span>
    <span class="edition">THE STUDY ROOM <i></i> VOL. 09</span></div>''')


def hero(compact=False):
    if compact:
        st.html('''<div class="conversation-heading"><span class="eyebrow">THE OPEN NOTEBOOK</span>
        <h1>One question leads to another.</h1></div>''')
        return
    _html('''<section class="notebook-hero">
    <div class="hero-copy"><div class="eyebrow"><span class="tiny-line"></span> A COMPANION FOR CURIOUS MINDS</div>
    <h1>Big ideas.<br>Little <em>aha!</em> moments.</h1>
    <p>From falling apples to the forces that move us.<br>Let’s make sense of physics, one question at a time.</p>
    <div class="hero-note"><span>↳</span> Your textbook. A fresh perspective.</div></div>
    <div class="orbital-art" aria-hidden="true">
    <span class="art-label">FIG. 01 — CURIOSITY IN MOTION</span>
    <svg viewBox="0 0 360 310" xmlns="http://www.w3.org/2000/svg">
      <defs><pattern id="grid" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".7" fill="#b5b2a7"/></pattern></defs>
      <rect x="15" y="8" width="330" height="290" fill="url(#grid)" opacity=".55"/>
      <path d="M30 155H330M180 15V290" stroke="#b8b3a8" stroke-width=".7" stroke-dasharray="3 7"/>
      <g class="atom-system">
        <ellipse cx="180" cy="155" rx="135" ry="49" fill="none" stroke="#4d5850" stroke-width="1.2" transform="rotate(-32 180 155)"/>
        <ellipse cx="180" cy="155" rx="135" ry="49" fill="none" stroke="#4d5850" stroke-width="1.2" transform="rotate(32 180 155)"/>
        <ellipse cx="180" cy="155" rx="135" ry="49" fill="none" stroke="#4d5850" stroke-width="1.2" transform="rotate(90 180 155)"/>
        <g transform="rotate(-32 180 155)"><g class="electron-track"><circle cx="0" cy="0" r="7" fill="#c35b3b"/><circle cx="0" cy="0" r="12" fill="none" stroke="#c35b3b" opacity=".25"/></g></g>
        <g transform="rotate(32 180 155)"><g class="electron-track second"><circle cx="0" cy="0" r="5" fill="#45584b"/></g></g>
        <g transform="rotate(90 180 155)"><g class="electron-track third"><circle cx="0" cy="0" r="5" fill="#c35b3b"/></g></g>
        <circle class="nucleus-halo" cx="180" cy="155" r="26" fill="#c35b3b" opacity=".08"/>
        <circle cx="180" cy="155" r="13" fill="#c35b3b"/><circle cx="176" cy="151" r="3" fill="#efb49b"/>
      </g>
      <path class="annotation-line" d="M205 168Q251 193 291 209" stroke="#a8a293" fill="none"/>
      <text x="277" y="230" fill="#77766c" font-size="15" font-family="Georgia, serif" font-style="italic">it starts here.</text>
      <text x="29" y="44" fill="#77766c" font-size="17" font-family="Georgia, serif" font-style="italic">F = ma</text>
      <path d="M29 50Q62 56 84 49" fill="none" stroke="#c35b3b" stroke-width="1"/>
    </svg><span class="art-footnote">A small question. An entire universe.</span>
    </div></section>''')


def section_heading():
    st.html('''<div class="section-heading"><div><span class="eyebrow">FOLLOW A SPARK</span>
    <h2>Where shall we begin?</h2></div><span>Pick a question, or bring your own <b>↙</b></span></div>''')


TOPICS = [
    ("01", "MOTION", "A change of pace.", "Speed & velocity", "What is the difference between speed and velocity?", "motion"),
    ("02", "FORCES", "Going in circles.", "Centripetal force", "What is centripetal force?", "force"),
    ("03", "GRAVITY", "Down to earth.", "Mass & weight", "How are mass and weight different?", "gravity"),
    ("04", "NEWTON’S LAWS", "Give. And take.", "Action & reaction", "Explain Newton's third law of motion.", "newton"),
]

SKETCHES = {
    "motion": '<path d="M10 48H120M15 33H39M22 24H46"/><circle cx="66" cy="35" r="13"/><path d="M91 35H126m-7-6 7 6-7 6"/>',
    "force": '<ellipse cx="68" cy="35" rx="36" ry="22" stroke-dasharray="3 5"/><circle cx="68" cy="35" r="3"/><path d="M68 35 99 23m-11 0 11 0-4 10"/><circle cx="99" cy="23" r="6" class="filled"/>',
    "gravity": '<path d="M20 58H118M68 6V41m-7-8 7 8 7-8"/><circle cx="68" cy="50" r="7" class="filled"/><path d="M36 58l-6 6m29-6-6 6m29-6-6 6m29-6-6 6"/>',
    "newton": '<circle cx="55" cy="35" r="13"/><circle cx="82" cy="35" r="13"/><path d="M38 35H10m7-6-7 6 7 6m81-6h28m-7-6 7 6-7 6"/>',
}


def topic_card(number, category, title, subtitle, sketch):
    _html(f'''<div class="topic-art topic-{sketch}"><div class="topic-index">{number}<span>{category}</span></div>
    <svg viewBox="0 0 140 72" aria-hidden="true">{SKETCHES[sketch]}</svg>
    <h3>{title}</h3><p>{subtitle}</p></div>''')


def footer(chunks):
    st.html(f'''<div class="notebook-footer"><span><i></i> GROUNDED IN YOUR TEXTBOOK</span>
    <span>{escape(str(chunks))} passages to learn from <b>·</b> Sources with every answer</span>
    <span class="footer-signature">Stay curious.</span></div>''')
