
function coneResponse = getConeResp(img_path)

    imageRGB = imread(img_path);
    fprintf('%d ',size(imageRGB)')
    
    tbUseProject('ISETImagePipeline');

    load("retina.mat", "retina")

    [~, ~, imageLinear, coneResponse] = retina.compute(double(imageRGB));

end 