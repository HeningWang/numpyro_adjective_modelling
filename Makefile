LATEX_DIR := 10-writing
LATEXMKRC := $(LATEX_DIR)/.latexmkrc
DRAFT_TEX := $(LATEX_DIR)/draft.tex
DRAFT_PDF := $(LATEX_DIR)/draft.pdf
BUILD_DIR := $(LATEX_DIR)/build

.PHONY: draft clean-draft

draft:
	mkdir -p $(BUILD_DIR)
	latexmk -cd -r $(LATEXMKRC) -pdf $(DRAFT_TEX)
	cp $(BUILD_DIR)/draft.pdf $(DRAFT_PDF)

clean-draft:
	latexmk -cd -r $(LATEXMKRC) -C $(DRAFT_TEX)
	rm -f $(DRAFT_PDF)
