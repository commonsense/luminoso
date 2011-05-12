import luminoso2
import divisi2
import os

def run_wiki_model(name, lang='en'):
    input_dir = 'inputs/'+name
    model_dir = 'models/'+name
    background = divisi2.network.conceptnet_assoc(lang)
    model = luminoso2.make_common_sense(model_dir, lang)
    del model

    for file in os.listdir(input_dir):
        model = luminoso2.load(model_dir)
        basename = file.rsplit('.', 1)[0]
        filename = (input_dir + os.sep + file).decode('utf-8')
        model.add_from(filename, study=basename.decode('utf-8'))
        del model
    model.recompute(background, {})
    return model
