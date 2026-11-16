"""references.bib for the thesis = references_core.bib + docs/related_work/references.bib,
with the research notes ('note' fields) turned into 'urldate' (Accessed ...) or dropped,
so they do not appear in the printed reference list."""
import re
from pathlib import Path
here = Path(__file__).resolve().parent
src = (here / "references_core.bib").read_text() + "\n" + (here.parent / "docs/related_work/references.bib").read_text()

def clean(entry):
    m = re.search(r"\n\s*note\s*=\s*\{(.*?)\},?\s*(?=\n)", entry, re.S)
    if not m:
        return entry
    acc = re.search(r"Accessed (\d{4}-\d{2}-\d{2})", m.group(1))
    repl = f"\n  urldate = {{{acc.group(1)}}}," if acc else ""
    return entry[:m.start()] + repl + entry[m.end():]

entries = re.findall(r"@\w+\{.*?\n\}", src, re.S)
seen, out = set(), []
for e in entries:
    key = re.match(r"@\w+\{([^,]+),", e).group(1).strip()
    if key in seen:
        continue
    seen.add(key)
    out.append(clean(e))
(here / "references.bib").write_text("\n\n".join(out) + "\n")
print(len(out), "entries")
