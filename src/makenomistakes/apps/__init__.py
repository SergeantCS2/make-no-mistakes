"""Workspace apps: OpenCode, LibreChat, Open WebUI, OpenHands, Goose and Odysseus. Each gets
one file (or card) built from the router's manifest (`mnm url --for <app> [--write]`, `card.py`
and one module per app) and a doctor run from where the app sits (`mnm doctor --app <app>`,
`doctor.py` over `probe.py`). mnm installs the first four (`mnm apps`: `hub.py` over one driver
per kind, the pins in `defs.py`) and connects the other two."""
