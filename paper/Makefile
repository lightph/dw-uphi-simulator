.PHONY: all pdf check-figs clean

MAIN = main
FIGS = fig1_dw_schematic.pdf fig2_rigid_wall_shapiro.pdf fig3_floquet_stability.pdf fig4_wall_metrics_ha3.pdf fig5_phase_diagram_L256.pdf fig6_edwards_wilkinson_scaling.pdf fig7_sp_phase_cascade.pdf fig8_sp_finite_size_scaling.pdf fig9_phase_diagram_L8.pdf

all: pdf

check-figs:
	@echo "Checking publication figures in figuras/..."
	@missing=0; \
	for f in $(FIGS); do \
		if [ ! -f "figuras/$$f" ]; then \
			echo "  [MISSING] figuras/$$f"; \
			missing=$$((missing + 1)); \
		else \
			echo "  [OK]      figuras/$$f"; \
		fi \
	done; \
	if [ $$missing -ne 0 ]; then \
		echo "ERROR: $$missing figure(s) missing. Run 'python3 reproduction/reproduce_all_figures.py --all --export-paper'"; \
		exit 1; \
	else \
		echo "All 9 publication figures are present."; \
	fi

pdf: check-figs
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf $(MAIN).tex; \
	else \
		pdflatex $(MAIN).tex && bibtex $(MAIN) && pdflatex $(MAIN).tex && pdflatex $(MAIN).tex; \
	fi

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk *Notes.bib

