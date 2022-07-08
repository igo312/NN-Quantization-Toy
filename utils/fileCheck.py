import os
import re
import pdb

def modelExist(ckpt_save_path, model_name):
    model_name = model_name + '.py'
    if not os.path.exists(os.path.join(ckpt_save_path, model_name)):
        return False
    else:
        return True

def getCkptPath(ckpt_save_path, model_name):
    pattern = re.compile(model_name+'.*?.pth')
    for filename in os.listdir(ckpt_save_path):
        #pdb.set_trace()
        if re.match(pattern, filename):
            return os.path.join(ckpt_save_path, filename)
    raise FileNotFoundError("The ckpt of {} is not founded".format(model_name))