import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from source.paper.plot_generator import PublicationPlotGenerator

def test_all_figures_generated_and_non_empty():
    generator = PublicationPlotGenerator()
    generator.generate_all_figures()
    
    figures = [
        "paper/figures/persistence_diagram.png",
        "paper/figures/pr_curves.png",
        "paper/figures/cycle_sensitivity.png",
        "paper/figures/scalability.png"
    ]
    
    for fig in figures:
        assert os.path.exists(fig), f"{fig} not found"
        assert os.path.getsize(fig) > 1000, f"{fig} is too small"

def test_plot_generator_headless_rendering():
    import matplotlib
    assert matplotlib.get_backend().lower() == 'agg'
