# Compile the paper
all: dual_sparse_architecture.pdf

dual_sparse_architecture.pdf: dual_sparse_architecture.tex
	pdflatex dual_sparse_architecture.tex
	pdflatex dual_sparse_architecture.tex  # Run twice for references

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg

distclean: clean
	rm -f *.pdf

.PHONY: all clean distclean
