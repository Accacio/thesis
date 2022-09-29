function print_image_pdf(filename,papersize)
% PRINT_IMAGE_PDF -
set(gcf, 'PaperPosition', [0 0 papersize])
set(gcf, 'PaperSize', papersize)
saveas(gcf,[ filename '.pdf'])
end
