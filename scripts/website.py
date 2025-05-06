import flask
import gensim
# import fasttext
import numpy as np


app = flask.Flask('API') # api = application interface (computers can chat w this server)
fasttext_model_path = 'data/cc.en.50.bin'
# ft_model = fasttext.load_model(fasttext_model_path)
gen_model = gensim.models.fasttext.load_facebook_vectors(fasttext_model_path)

@app.route('/') 
def heartbeat():
    return flask.jsonify({'alive': True})

@app.route('/math', methods = ['GET'])
def do_math():
    number = int(flask.request.args.get('number'))
    return flask.jsonify({'status': 'complete', 'number': number*10})

@app.route('/thesarus', methods = ['GET'])
def thes():
    return flask.render_template('thesarus.html')


@app.route('/sentence', methods = ['GET'])
def sentence():
    input_sentence = flask.request.args.get('sentence')
    if (input_sentence and len(input_sentence) >0):
        input_sentence =[v.lower() for v in input_sentence.split('_')]
        model_vectors = [gen_model.get_vector(v) for v in input_sentence]
        # av_vector = 



if __name__ == '__main__':
    app.run(port = 8000)
