all:	main.tex
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex
	zathura main.pdf

abstract:
	pdflatex main.tex
	zathura main.pdf

clean:
	rm ./main.aux ./main.log ./main.out ./main.pdf
