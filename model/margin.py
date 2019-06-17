from model.margin_backbone.ArcMarginProduct import ArcMarginProduct
from model.margin_backbone.CosineMarginProduct import CosineMarginProduct
from model.margin_backbone.InnerProduct import InnerProduct
from model.margin_backbone.SphereMarginProduct import SphereMarginProduct


def margin_select(opt):
    if opt.margin_type == 'ArcFace':
        margin = ArcMarginProduct(opt.embedding_size, opt.out_size, s=opt.margin_s)
    elif opt.margin_type == 'CosFace':
        margin = CosineMarginProduct(opt.embedding_size, opt.out_size, s=opt.margin_s)
    elif opt.margin_type == 'Softmax':
        margin = InnerProduct(opt.embedding_size, opt.out_size)
    elif opt.margin_type == 'SphereFace':
        margin = SphereMarginProduct(opt.embedding_size, opt.out_size)
    else:
        print(opt.margin_type, 'is not available!')
    return margin
        