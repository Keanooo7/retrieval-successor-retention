---
id: R-2026-09-22-inert-checkpoint-quarantine
date: 2026-09-22
stated_in: interactive planning session, 2026-09-22 ("Write, quarantined")
---
An inert run still writes its checkpoint, but under `runs/<id>/quarantine/` with an
`INERT` marker file. Loaders refuse a quarantined checkpoint unless explicitly
overridden. Keeps forensics; nothing downstream consumes it silently.
