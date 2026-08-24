import re
import os
import subprocess

class PaperCompiler:
    def __init__(self, tex_file="paper/main.tex", bib_file="paper/references.bib"):
        self.tex_file = tex_file
        self.bib_file = bib_file
    
    def validate_latex_syntax(self):
        with open(self.tex_file, 'r') as f:
            content = f.read()
        environments = re.findall(r'\\(begin|end)\{([^\}]+)\}', content)
        stack = []
        for tag, name in environments:
            if tag == 'begin':
                stack.append(name)
            elif tag == 'end':
                if not stack or stack[-1] != name:
                    return False
                stack.pop()
        return len(stack) == 0

    def check_balanced_environments(self):
        return self.validate_latex_syntax()
    
    def verify_cite_keys(self):
        with open(self.tex_file, 'r') as f:
            tex_content = f.read()
        with open(self.bib_file, 'r') as f:
            bib_content = f.read()
            
        cites = set(re.findall(r'\\cite\{([^\}]+)\}', tex_content))
        bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib_content))
        
        for cite in cites:
            for key in cite.split(','):
                if key.strip() not in bib_keys:
                    return False
        return True
        
    def compile_pdf(self):
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", self.tex_file], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
