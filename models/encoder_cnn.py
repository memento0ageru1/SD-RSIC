from torch.nn import Module, AdaptiveAvgPool2d, Linear, BatchNorm1d, Sequential, Flatten
import torchvision

class EncoderCNN(Module):
    def __init__(self, embed_size=14, model_name='resnet152'):
        super().__init__()
        self.model, out_features = eval('self.build_' + model_name + '_module()')
        self.linear = Linear(out_features, embed_size).to('cuda', non_blocking=True)
        self.batch_norm = BatchNorm1d(embed_size, momentum=0.01).to('cuda', non_blocking=True)
        
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        out = self.model(images)
        out = out.reshape(out.size(0), -1)
        out = self.batch_norm(self.linear(out))    
        return out

    def build_resnet34_module(self):
        resnet = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]
        return Sequential(*modules).to('cuda', non_blocking=True), resnet.fc.in_features

    def build_resnet50_module(self):
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        return Sequential(*modules).to('cuda', non_blocking=True), resnet.fc.in_features

    def build_resnet101_module(self):
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        return Sequential(*modules).to('cuda', non_blocking=True), resnet.fc.in_features

    def build_resnet152_module(self):
        resnet = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        return Sequential(*modules).to('cuda', non_blocking=True), resnet.fc.in_features

    def build_vgg16_module(self):
        model = torchvision.models.vgg16(pretrained=True)
        feat_modules = Sequential(*list(model.features.children())).to('cuda', non_blocking=True)
        cls_modules = Sequential(*list(model.classifier.children())[:1]).to('cuda', non_blocking=True)
        return Sequential(feat_modules, AdaptiveAvgPool2d((7, 7)), Flatten(), cls_modules).to('cuda', non_blocking=True), model.classifier[0].out_features

    def build_vgg19_module(self):
        model = torchvision.models.vgg19(pretrained=True)
        feat_modules = Sequential(*list(model.features.children())).to('cuda', non_blocking=True)
        cls_modules = Sequential(*list(model.classifier.children())[:1]).to('cuda', non_blocking=True)
        return Sequential(feat_modules, AdaptiveAvgPool2d((7, 7)), Flatten(), cls_modules).to('cuda', non_blocking=True), model.classifier[0].out_features

    def build_inceptionv1_module(self):
        model = torchvision.models.googlenet(pretrained=True)
        modules = list(model.children())[:-1]
        return Sequential(*modules).to('cuda', non_blocking=True), model.fc.in_features

    def build_inceptionv3_module(self):
        model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        modules = Sequential(*list(model.children())[:-1])
        return Sequential(modules, AdaptiveAvgPool2d((1, 1)), Flatten()).to('cuda', non_blocking=True), model.fc.in_features

    def build_densenet121_module(self):
        model = torchvision.models.densenet121(pretrained=True)
        feat_modules = Sequential(*list(model.features.children())).to('cuda', non_blocking=True)
        cls_modules = Sequential(*list(model.classifier.children())[:-1]).to('cuda', non_blocking=True)
        return Sequential(feat_modules, AdaptiveAvgPool2d((1, 1)), Flatten(), cls_modules).to('cuda', non_blocking=True), model.classifier.in_features
    
    def build_densenet169_module(self):
        model = torchvision.models.densenet169(pretrained=True)
        feat_modules = Sequential(*list(model.features.children())).to('cuda', non_blocking=True)
        cls_modules = Sequential(*list(model.classifier.children())[:-1]).to('cuda', non_blocking=True)
        return Sequential(feat_modules, AdaptiveAvgPool2d((1, 1)), Flatten(), cls_modules).to('cuda', non_blocking=True), model.classifier.in_features

    def build_densenet201_module(self):
        model = torchvision.models.densenet201(pretrained=True)
        feat_modules = Sequential(*list(model.features.children())).to('cuda', non_blocking=True)
        cls_modules = Sequential(*list(model.classifier.children())[:-1]).to('cuda', non_blocking=True)
        return Sequential(feat_modules, AdaptiveAvgPool2d((1, 1)), Flatten(), cls_modules).to('cuda', non_blocking=True), model.classifier.in_features