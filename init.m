function init()
    disp("---Installing ISETImagePipeline for this engine---")
    tbUseProject('ISETImagePipeline');
    
    disp('--- Creating retina ---')
    retina = ConeResponse('eccBasedConeDensity', true, 'eccBasedConeQuantal', true, ...
        'fovealDegree', 1.0, 'pupilSize', 2.5);
    disp('---Saving retina ---')
    save("retina.mat", "retina")
end 