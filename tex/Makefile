.PHONY: all viewpdf pdf clean clean-pdf

TARGET       = main
SOURCE_FILES = $(TARGET).tex $(wildcard */*.tex)
BIB_FILES    = $(wildcard biblio/*.bib)
FIGURES      = $(wildcard */figures/*)

# Set the pdf reader according to the operating system
OS = $(shell uname)
ifeq ($(OS), Darwin)
	PDF_READER = open
endif
ifeq ($(OS), Linux)
	PDF_READER = xdg-open
endif

all: pdf

viewpdf: pdf
	$(PDF_READER) $(TARGET).pdf &

pdf: $(TARGET).pdf

$(TARGET).pdf: $(SOURCE_FILES) $(BIB_FILES) $(FIGURES) these-dbl.cls
	pdflatex -interaction=nonstopmode -jobname=$(TARGET) $(SOURCE_FILES)
	makeglossaries $(TARGET)
	biber $(TARGET)
	pdflatex -interaction=nonstopmode -synctex=15 -jobname=$(TARGET) $(SOURCE_FILES)
	pdflatex -interaction=nonstopmode -synctex=15 -jobname=$(TARGET) $(SOURCE_FILES)

clean:
	@git clean -dfX
clean-pdf:
	@rm -f $(TARGET).pdf

preview: $(TARGET).pdf
	convert -density 300 'main.pdf[0]' ../main_preview.png
