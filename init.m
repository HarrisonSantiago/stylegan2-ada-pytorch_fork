function = init(home_dir)
    disp("---Installing ISETImagePipeline for this engine---")
    tbUseProject('ISETImagePipeline');
    
    disp('--- Creating retina ---')
    retina = ConeResponse('eccBasedConeDensity', true, 'eccBasedConeQuantal', true, ...
        'fovealDegree', 1.0, 'pupilSize', 2.5);
    disp('---Saving retina ---')
    save(home_dir + "/retina.mat", "retina")
end 