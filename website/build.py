"""Inline data.json into index.src.html -> index.html"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "index.src.html"), encoding="utf-8").read()
data = open(os.path.join(HERE, "data.json"), encoding="utf-8").read()
assert "/*__DATA__*/" in src
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8", newline="\n").write(src.replace("/*__DATA__*/", data))
print("wrote website/index.html")
