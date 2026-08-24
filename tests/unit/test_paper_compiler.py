import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from source.paper.compiler import PaperCompiler

def test_latex_document_syntax_and_structure():
    compiler = PaperCompiler()
    assert compiler.validate_latex_syntax() == True
    assert compiler.check_balanced_environments() == True
    
    with open("paper/main.tex", "r") as f:
        content = f.read()
        
    required_sections = [
        "Abstract", "Introduction", "Related Work", "Topological Formulation",
        "Architecture (TopoRingNet)", "Experimental Evaluation", 
        "Results \\& Pre-Registered Hypotheses H1-H4", "Discussion", "Conclusion"
    ]
    for section in required_sections:
        assert section.lower().replace("\\", "") in content.lower().replace("\\", ""), f"Section {section} missing"

def test_bibliography_keys_resolved():
    compiler = PaperCompiler()
    assert compiler.verify_cite_keys() == True
