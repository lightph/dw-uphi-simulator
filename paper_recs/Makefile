# ==============================================================================
# Makefile for compiling the manuscript and supplementary material
# ==============================================================================

MAIN = main
SUPPL = supplement
LATEX = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

.PHONY: all clean distclean check-figs help

all: $(MAIN).pdf $(SUPPL).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib $(wildcard figuras/*.pdf)
	@echo "==> Pass 1: pdflatex $(MAIN)..."
	@$(LATEX) $(MAIN).tex > /dev/null || (cat $(MAIN).log && false)
	@echo "==> Running bibtex $(MAIN)..."
	@$(BIBTEX) $(MAIN) > /dev/null || (cat $(MAIN).blg && false)
	@echo "==> Pass 2: pdflatex $(MAIN)..."
	@$(LATEX) $(MAIN).tex > /dev/null
	@echo "==> Pass 3: pdflatex $(MAIN)..."
	@$(LATEX) $(MAIN).tex > /dev/null
	@echo "==> Built $(MAIN).pdf successfully."

$(SUPPL).pdf: $(SUPPL).tex
	@echo "==> Building $(SUPPL).pdf..."
	@$(LATEX) $(SUPPL).tex > /dev/null
	@$(LATEX) $(SUPPL).tex > /dev/null
	@echo "==> Built $(SUPPL).pdf successfully."

check-figs:
	@echo "==> Checking required publication figures in figuras/:"
	@for fig in snap_system_345.pdf figure2.pdf figure3.pdf sweep_ha_3.pdf \
	            phasediagram_L256.pdf NC_EW_inset.pdf sc_cascade.pdf \
	            SC_scaling.pdf phasediagram_L8.pdf; do \
		if [ -f "figuras/$$fig" ]; then \
			echo "  [OK] figuras/$$fig"; \
		else \
			echo "  [MISSING] figuras/$$fig"; \
		fi \
	done

clean:
	@echo "==> Cleaning auxiliary LaTeX build files..."
	@rm -f *.aux *.log *.out *.toc *.bbl *.blg *Notes.bib *.synctex.gz *.fls *.fdb_latexmk
	@echo "==> Done."

distclean: clean
	@echo "==> Removing generated PDF files..."
	@rm -f $(MAIN).pdf $(SUPPL).pdf
	@echo "==> Done."

help:
	@echo "Available make targets:"
	@echo "  make              Build main.pdf and supplement.pdf"
	@echo "  make check-figs   Verify all 9 publication figures exist"
	@echo "  make clean        Remove auxiliary LaTeX build files"
	@echo "  make distclean    Remove build files and generated PDFs"

