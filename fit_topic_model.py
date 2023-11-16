import argparse
from json import load
from datasets import load_from_disk
import tomotopy as tp

import nltk
from nltk.corpus import stopwords

def prepare_corpus(data_filename):
    doc_dataset = load_from_disk(f'data/{data_filename}')
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    english_stops = set(stopwords.words('english'))

    def tokenize(text):
        return [token for token in text.lower().split() if token.isalpha()]
    
    preprocessed_text = [tokenize(text) for text in doc_dataset['text']]
    corpus = tp.utils.Corpus(stopwords=lambda x: x in english_stops)

    for doc in preprocessed_text:
        corpus.add_doc(doc)
        
    return corpus

def fit_model(corpus, args, print_n_top=10, verbose_print=False):
    if args.model_type == 'CTM':
        mdl = tp.CTModel(
                tw=tp.TermWeight.IDF,
                min_cf=args.min_cf,
                min_df=args.min_df, 
                rm_top=args.rm_top, 
                k=args.num_topics, 
                corpus=corpus
                )
    else:
        print('Model type not supported')
        return None

    mdl.train(0)
    mdl.num_beta_sample = args.num_beta_sample

    print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
        len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
    ))
    print('Removed Top words: ', *mdl.removed_top_words)

    for i in range(0, args.iters, 10):
        try:
            mdl.train(10)
            print('Iteration: {:04}, LL per word: {:.4}'.format(i, mdl.ll_per_word))
        except Exception as e:
            print(f"An error occurred at iteration {i}: {e}")
            break

    for k in range(mdl.k):
        if verbose_print:
            print('Topic #{}: {}'.format(k, mdl.get_topic_words(k, top_n=print_n_top)))
        else:
            label = "#{}".format(k)
            title= ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=6))
            print('Topic', label, title)
    return mdl

def main(args):
    corpus = prepare_corpus(args.data_filename)
    mdl = fit_model(corpus, args)
    if not mdl:
        return

    model_name_parts = []
    for arg, value in vars(args).items():
        if arg not in ['data_filename', 'model_type', 'num_beta_sample']:
            model_name_parts.append(f"{arg}")
        model_name_parts.append(f"{value}")
    model_name = '-'.join(model_name_parts)
    
    print(f"Saving model: {model_name}.bin")
    mdl.save(f'models/{model_name}.bin')    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default='wiki_math_subset')
    parser.add_argument('--model_type', type=str, default='CTM')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--min_cf', type=int, default=0)
    parser.add_argument('--min_df', type=int, default=0)
    parser.add_argument('--rm_top', type=int, default=0)
    parser.add_argument('--num_topics', type=int, default=10)
    parser.add_argument('--num_beta_sample', type=int, default=10)
    args = parser.parse_args()
    main(args)