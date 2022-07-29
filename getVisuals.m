function getVisuals(retinaPath, imPath)

    %modified visualExcitation and visualizeOI so that they would return
    %the frame
    
    %load(retinaPath, "retina")

    global stored_retina;
    retina = stored_retina;

    imageRGB = imread(imPath);
    
    [~, ~, ~, ~] = retina.compute(double(imageRGB));
    visexc = retina.visualizeExcitation();
    visOI = retina.visualizeOI();
  
    F = getframe(visexc);
    [im, ~] = frame2im(F);
    
    F1 = getframe(visOI);
    [im1, ~] = frame2im(F1);
    
    h=montage({im, im1});
    montage_IM=h.CData;
    %write to file
    imwrite(montage_IM,'for_mp4.png');

end