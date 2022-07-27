
function coneResponse = getConeResp(img_path, retina_path)

    imageRGB = imread(img_path);

    load(retina_path, "retina")

    [~, ~, imageLinear, coneResponse] = retina.compute(double(imageRGB));

end 