function init(home_dir, im_width)
    disp("---Installing ISETImagePipeline for this engine---")
    tbUseProject('ISETImagePipeline');


    name = strcat(home_dir, "/retina",im_width,".mat");
    if exist(name, 'file')
        disp('retina, render, and pinv_render exist for this size, loading retina...')
        name = strcat(home_dir, "/render",im_width,".mat");
        retina = load(name, "retina");
    else
        disp('--- Creating retina ---')
        retina = ConeResponse('eccBasedConeDensity', true, 'eccBasedConeQuantal', true, ...
            'fovealDegree', 1.0, 'pupilSize', 2.5);
        disp('---Saving retina ---')
        save(name, "retina");

        disp('---Saving render mat ---')
        imageSize = [32,32,3] %TODO: make able to modify
        render = retina.forwardRender(imageSize);
        render = double(render);
        name = strcat(home_dir, "/render",im_width,".mat");
        save(name, "render");

        disp('--- Saving render pinv ---')
        render_pinv = pinv(render, 1e-4);
        name = strcat(home_dir, "/render_pinv", im_width, ".mat");
        save(name, "render_pinv");
    end

    disp('--- creating global retina variable ---')
    global stored_retina;
    stored_retina = retina;

end 