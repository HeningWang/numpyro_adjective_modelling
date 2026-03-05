# .latexmkrc — latexmk configuration for this handout
# Use pdflatex with shell-escape
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$pdf_mode = 1;

# Re-run until stable (resolves "Label may have changed" and rerunfilecheck warnings)
$max_repeat = 5;
