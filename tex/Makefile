clean:
	ls | grep main | grep -v main.tex | xargs rm -rf
	ls | grep minted | xargs rm -rf

main.pdf: main.tex reference.bib
	pdflatex -shell-escape main && bibtex main && pdflatex -shell-escape main && pdflatex -shell-escape main

spell-check:
	@cat *.tex | aspell list --dont-backup --lang=en --mode=tex --personal ./.aspell.en.pws --add-tex-command "citep op" --add-tex-command "citet op" --add-tex-command "cite op" --add-tex-command "bibliographystyle op" --add-tex-command "author op" | sort | uniq

.PHONY: clean spell-check
