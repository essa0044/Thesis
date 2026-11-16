"""Scale every table grid (fixed layout) in Thesis.docx to the text width of the A4 page (pandoc assumes 7920 twips)."""
import re, sys, zipfile, shutil
path = sys.argv[1]
z = zipfile.ZipFile(path)
doc = z.read("word/document.xml").decode()
pg = re.search(r'<w:pgSz w:h="\d+" w:w="(\d+)"', doc)
mar = re.search(r'<w:pgMar[^>]*w:left="(\d+)"[^>]*w:right="(\d+)"', doc)
width = int(pg.group(1)) - int(mar.group(1)) - int(mar.group(2))

def scale(m):
    cols = [int(x) for x in re.findall(r'<w:gridCol w:w="(\d+)"', m.group(0))]
    tot = sum(cols) or 1
    return "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{round(c * width / tot)}" />' for c in cols) + "</w:tblGrid>"

doc = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", scale, doc, flags=re.S)
# fixed layout, so viewers keep the computed column widths instead of auto-fitting
doc = doc.replace('<w:tblW w:type="pct" w:w="5000" />', '<w:tblW w:type="pct" w:w="5000" /><w:tblLayout w:type="fixed" />')
tmp = path + ".tmp"
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
    for item in z.infolist():
        if item.filename.endswith("/"):
            continue
        data = doc.encode() if item.filename == "word/document.xml" else z.read(item.filename)
        out.writestr(item, data)
z.close()
shutil.move(tmp, path)
