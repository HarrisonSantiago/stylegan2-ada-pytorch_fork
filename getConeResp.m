
function coneResponse = getConeResp(img_path, retina_path)

    imageRGB = imread(img_path);
    fprintf('%d ',size(imageRGB)')

    load(retina_path, "retina")

    [~, ~, imageLinear, coneResponse] = retina.compute(double(imageRGB));

end 