LATEX_DIR := paper
LATEXMKRC := $(LATEX_DIR)/.latexmkrc
DRAFT_TEX := $(LATEX_DIR)/draft.tex
DRAFT_PDF := $(LATEX_DIR)/draft.pdf
BUILD_DIR := $(LATEX_DIR)/build
DRAFT_AUX := $(addprefix $(LATEX_DIR)/draft.,aux bbl bcf blg fdb_latexmk fls log out run.xml synctex.gz toc)

.PHONY: draft clean clean-draft

draft:
	mkdir -p $(BUILD_DIR)
	latexmk -g -cd -r $(LATEXMKRC) -pdf $(DRAFT_TEX)
	cp $(BUILD_DIR)/draft.pdf $(DRAFT_PDF)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(DRAFT_AUX)

clean-draft: clean
	rm -f $(DRAFT_PDF)
