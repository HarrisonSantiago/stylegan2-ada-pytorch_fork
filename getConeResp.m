
function coneResponse = getConeResp(img_path)

    imageRGB = imread(img_path);
    fprintf('%d ',size(imageRGB)')
    
    tbUseProject('ISETImagePipeline');
    %display = displayCreate('CRT12BitDisplay');

    retina = ConeResponse('eccBasedConeDensity', true, 'eccBasedConeQuantal', true, ...
        'fovealDegree', 1.0, 'pupilSize', 2.5);

    %load("render.mat", "render")

    [~, ~, imageLinear, coneResponse] = retina.compute(double(imageRGB));
    %coneResponse = render * imageLinear(:);
end 