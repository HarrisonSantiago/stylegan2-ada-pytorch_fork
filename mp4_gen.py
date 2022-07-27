from PIL import Image

def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_h_multi_blank(im_list):
    #joins images along horizontal axis when they have different heights
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_h_blank(_im, im)
    return _im

#example of how to call
#get_concat_h_multi_blank([im1, im2, im1]).save('data/dst/pillow_concat_h_multi_blank.jpg')
