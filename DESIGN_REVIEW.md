# Physics Notebook — design review

The redesign was reviewed and approved for a GitHub push on September 22, 2026.

## Preview

Run `.venv/bin/python -m streamlit run app.py`, then open http://localhost:8501.
Screenshots are in `design-preview/desktop.png`, `design-preview/mobile.png`,
and `design-preview/textbook.png`. Open the running app to review motion.

## Direction

Warm paper, forest-green book cloth, rust accents, editorial serif headings,
and small scientific sketches. The opening screen invites a question; model and
retrieval controls live in sidebar expanders. The existing grounded answers,
source excerpts, textbook search, retry action, and transcript download remain available.

Motion includes orbiting electrons, a breathing nucleus, staggered card entrances,
card and book hover movement, input focus transitions, and new-message entrances.
The app honors the operating system's reduced-motion preference. On phones, the
sidebar starts collapsed and the question cards form two columns.

Illustrations use local SVG data, with no image service or JavaScript dependency.
Google Fonts supplies Instrument Serif, DM Sans, and DM Mono; system fonts are
used if that service is unavailable.

## Validation

```sh
.venv/bin/python -m unittest test_answer_book.py test_notebook_ui.py -q
```

UI checks cover the notebook and textbook tabs, rendering and escaping source
excerpts, resetting a conversation, and submitting a starter question with a
mocked answer service. Browser checks cover desktop/mobile rendering, illustration
loading, tab switching, animation, and reduced motion. Live Groq generation was
not called during this design review.

Presentation lives in `notebook_ui.py` and `assets/notebook.css`; `app.py` connects
the design to the existing application. Streamlit styling uses its DOM test IDs
and custom container keys, so review the UI when upgrading Streamlit.
