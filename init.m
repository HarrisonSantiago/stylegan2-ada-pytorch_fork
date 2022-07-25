function init(home_dir, im_width)
    disp("---Installing ISETImagePipeline for this engine---")
    tbUseProject('ISETImagePipeline');

    save(home_dir + "/retina.mat", "retina")


    name = strcat(home_dir, "/retina",im_width,".mat");
    if exist(name, 'file')
        disp('retina, render, and pinv_render exist for this size')
    else
        disp('--- Creating retina ---')
        retina = ConeResponse('eccBasedConeDensity', true, 'eccBasedConeQuantal', true, ...
            'fovealDegree', 1.0, 'pupilSize', 2.5);
        disp('---Saving retina ---')
        save(name, "retina");

        disp('---Saving render mat ---')
        render = retina.forwardRender(imageSize);
        render = double(render);
        name = strcat(home_dir, "/render",im_width,".mat");
        save(name, "render");

        disp('--- Saving render pinv ---')
        render_pinv = pinv(render, 1e-4);
        name = strcat(home_dir, "/render_pinv", im_width, ".mat");
        save(name, "render_pinv");
    end

end 