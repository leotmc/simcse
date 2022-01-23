from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import io

from senteval.tools.validation import KFoldClassifier

from sklearn.metrics import f1_score


class BQEval(object):
    def __init__(self, task_path, tokenizer, seed=1111):
        logging.info('***** Transfer task : BQ *****\n\n')
        self.seed = seed
        self.tokenizer = tokenizer
        train = self.loadFile(os.path.join(task_path,
                              'BQ.train.data'))
        test = self.loadFile(os.path.join(task_path,
                             'BQ.train.data'))
        self.bq_data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.bq_data['train']['X_A'] + \
                  self.bq_data['train']['X_B'] + \
                  self.bq_data['test']['X_A'] + self.bq_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        bq_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                bq_data['X_A'].append(self.tokenizer.tokenize(text[0]))
                bq_data['X_B'].append(self.tokenizer.tokenize(text[1]))
                bq_data['y'].append(text[2])
        # bq_data['X_A'] = bq_data['X_A'][:1000]
        # bq_data['X_B'] = bq_data['X_B'][:1000]
        # bq_data['y'] = bq_data['y'][:1000]
        bq_data['y'] = [int(s) for s in bq_data['y'][1:]]
        return bq_data

    def run(self, params, batcher):
        bq_embed = {'train': {}, 'test': {}}

        for key in self.bq_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.bq_data[key]['X_A'],
                                       self.bq_data[key]['X_B'],
                                       self.bq_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            text_data['A'] = [x for (x, y, z) in sorted_corpus]
            text_data['B'] = [y for (x, y, z) in sorted_corpus]
            text_data['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['A', 'B']:
                bq_embed[key][txt_type] = []
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    bq_embed[key][txt_type].append(embeddings)
                bq_embed[key][txt_type] = np.vstack(bq_embed[key][txt_type])
            bq_embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = bq_embed['train']['A']
        trainB = bq_embed['train']['B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = bq_embed['train']['y']

        # Test
        testA = bq_embed['test']['A']
        testB = bq_embed['test']['B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = bq_embed['test']['y']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = KFoldClassifier(train={'X': trainF, 'y': trainY},
                              test={'X': testF, 'y': testY}, config=config)

        devacc, testacc, yhat = clf.run()
        testf1 = round(100*f1_score(testY, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for bq.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainA), 'ntest': len(testA)}
