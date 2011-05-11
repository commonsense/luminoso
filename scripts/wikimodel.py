import luminoso2
import os

def run_wiki_model(name, lang='en'):
    input_dir = 'inputs/'+name
    model_dir = 'models/'+name
    model = luminoso2.make_common_sense(model_dir, lang)

    for file in os.listdir(input_dir):
        basename = file.rsplit('.', 1)[0]
        filename = (input_dir + os.sep + file).decode('utf-8')
        model.learn_from(filename, study=basename.decode('utf-8'))

