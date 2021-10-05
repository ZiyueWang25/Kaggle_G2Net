from .models_1d import *
from .models_2d import *
from .models_3d import *


def M1D(model_dict):  
    if model_dict['model_module'] == "V2":
        model = ModelIafossV2(n=model_dict['channels'],
                      use_raw_wave=model_dict['use_raw_wave'],sdrop=model_dict['sdrop'])
                              
    elif model_dict['model_module']  == 'V2SD':
        model = V2StochasticDepth(n=model_dict['channels'],
                      proba_final_layer=model_dict['proba_final_layer'], 
                      use_raw_wave=model_dict['use_raw_wave'],sdrop=model_dict['sdrop'])  
                              
    elif model_dict['model_module'] == "V2S":
        model = ModelIafossV2S(n=model_dict['channels'],
                      use_raw_wave=model_dict['use_raw_wave'],sdrop=model_dict['sdrop'])
        
    return model

def M2D(model_dict):
    if model_dict['model_module'] == 'resnet34':
        model = Model_2D(encoder=model_dict["encoder"], 
                         use_raw_wave=model_dict['use_raw_wave']
                        )
    return model

def M3D(model_dict):
    if model_dict['model_module'] == "M3D":
        fold = model_dict['fold']
        
        model_dict['model_module'] = model_dict['model_1D']
        model_1d = M1D(model_dict)
        if model_dict['model_1D_pretrain_dir'] is not None and model_dict['fold'] is not None:
            path = f"{model_dict['model_1D_pretrain_dir']}/Fold_{fold}_best_model.pth"
            print("Loading model from path: ", path)        
            checkpoint = torch.load(path, map_location='cuda:0')
            model_1d.load_state_dict(checkpoint['model_state_dict'])
        model_1d.use_raw_wave=False
        
        model_dict['model_module'] = model_dict['model_2D']
        model_2d = M2D(model_dict)
        if model_dict['model_2D_pretrain_dir'] is not None and model_dict['fold'] is not None:
            path = f"{model_dict['model_2D_pretrain_dir']}/Fold_{fold}_best_model.pth"
            print("Loading model from path: ", path)        
            checkpoint = torch.load(path, map_location='cuda:0')
            model_2d.load_state_dict(checkpoint['model_state_dict'])
        model_2d.use_raw_wave=False
        
        model = Combined1D2D(model_1d, model_2d, 
                             emb_1d=model_dict['model_1D_emb'], 
                             emb_2d=model_dict['model_2D_emb'],
                             first=model_dict['first'], 
                             ps=model_dict['ps'])
        model.freeze_conv(req_grad=False)
        model_dict['model_module'] = "M3D"
    return model
        
        

def Model(model_dict):
    if model_dict['model_module'] in ['resnet34']:
        return M2D(model_dict)
    elif model_dict['model_module'] in ['M3D']:
        return M3D(model_dict)
    else:
        return M1D(model_dict)

