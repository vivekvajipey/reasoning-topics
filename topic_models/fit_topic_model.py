import argparse
from json import load
from datasets import load_from_disk
import tomotopy as tp
import numpy as np
import time

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

def fit_model(corpus, args, log_filename, print_n_top=10, verbose_print=False):
    try:
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
            with open(log_filename, 'a') as log_file:
                log_file.write('Model type not supported\n')
            return None

        mdl.train(0)
        mdl.num_beta_sample = args.num_beta_sample

        with open(log_filename, 'a') as log_file:
            log_file.write(f'Num docs:{len(mdl.docs)}, Num Vocabs:{len(mdl.used_vocabs)}, Total Words:{mdl.num_words}\n')
            log_file.write('Removed Top words: ' + ' '.join(mdl.removed_top_words) + '\n')

            num_skipped = 0
            for i in range(0, args.iters, 10):
                try:
                    mdl.train(10)
                    ll_per_word = mdl.ll_per_word
                    if not np.isfinite(ll_per_word):
                        raise ValueError(f"Non-finite log-likelihood per word at iteration {i}: {ll_per_word}")
                    log_file.write(f'Iteration: {i:04}, LL per word: {ll_per_word:.4}\n')
                except ValueError as e:
                    log_file.write(f"ValueError encountered: {e}\n")
                    num_skipped += 10
                    continue
                except Exception as e:
                    log_file.write(f"An unexpected error occurred at iteration {i}: {e}\n")
                    num_skipped += 10
                    continue

            log_file.write(f'Documents skipped: {num_skipped}\n')

            for k in range(mdl.k):
                if verbose_print:
                    topic_words = 'Topic #{}: {}'.format(k, mdl.get_topic_words(k, top_n=print_n_top))
                else:
                    label = "#{}".format(k)
                    title= ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=6))
                    topic_words = 'Topic ' + label + ' ' + title
                log_file.write(topic_words + '\n')

        return mdl

    except Exception as e:
        with open(log_filename, 'a') as log_file:
            log_file.write(f'Critical error during model training: {e}\n')
        return None

def main(args):
    start_time = time.time()
    corpus = prepare_corpus(args.data_filename)
    
    model_name_parts = []
    for arg, value in vars(args).items():
        if arg not in ['data_filename', 'model_type', 'num_beta_sample']:
            model_name_parts.append(f"{arg}")
        model_name_parts.append(f"{value}")
    model_name = '-'.join(model_name_parts)
    log_filename = f"logs/{model_name}.log"

    mdl = fit_model(corpus, args, log_filename)
    if not mdl:
        return    
    
    print(f"Saving model: {model_name}.bin")
    mdl.save(f'models/{model_name}.bin')
    print('Total time: ', time.time() - start_time)

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