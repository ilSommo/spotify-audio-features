pdf:
	pdflatex --shell-escape report.tex
	biber report
	pdflatex --shell-escape report.tex
	pdflatex --shell-escape report.tex

all:
	latexindent report.tex -o report.tex
	pdflatex --shell-escape report.tex
	biber report
	pdflatex --shell-escape report.tex
	pdflatex --shell-escape report.tex
	rm *.aux *.bbl *.bcf *.blg *.lof *.log *.out *run.xml *.toc
	rm -r _minted-report