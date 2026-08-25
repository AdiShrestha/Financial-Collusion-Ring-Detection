import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from source.paper.compiler import PaperCompiler

def test_latex_document_syntax_and_structure():
    compiler = PaperCompiler(tex_file="paper/kuset_main.tex")
    assert compiler.validate_latex_syntax() == True
    assert compiler.check_balanced_environments() == True
    
    with open("paper/kuset_main.tex", "r") as f:
        content = f.read()
        
    required_sections = [
        "Abstract", "Introduction", "Materials and Methods", "Results",
        "Discussion", "Conclusion", "Author and Data-Availability Note"
    ]
    for section in required_sections:
        assert section.lower().replace("\\", "") in content.lower().replace("\\", ""), f"Section {section} missing"

def test_bibliography_keys_resolved():
    with open("paper/kuset_main.tex", "r", encoding="utf-8") as handle:
        content = handle.read()
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", content)
        for key in group.split(",")
    }
    embedded = set(re.findall(r"\\bibitem\{([^}]+)\}", content))
    assert cited
    assert cited <= embedded
