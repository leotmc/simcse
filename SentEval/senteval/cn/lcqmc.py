from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import io

from senteval.tools.validation import KFoldClassifier

from sklearn.metrics import f1_score


class LCQMCEval(object):
    def __init__(self, task_path, tokenizer, seed=1111):
        logging.info('***** Transfer task : LCQMC *****\n\n')
        self.seed = seed
        self.tokenizer = tokenizer
        train = self.loadFile(os.path.join(task_path,
                              'LCQMC.train.data'))
        test = self.loadFile(os.path.join(task_path,
                             'LCQMC.test.data'))
        self.lcqmc_data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.lcqmc_data['train']['X_A'] + \
                  self.lcqmc_data['train']['X_B'] + \
                  self.lcqmc_data['test']['X_A'] + self.lcqmc_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        lcqmc_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                lcqmc_data['X_A'].append(self.tokenizer.tokenize(text[0]))
                lcqmc_data['X_B'].append(self.tokenizer.tokenize(text[1]))
                lcqmc_data['y'].append(text[2])
        # lcqmc_data['X_A'] = lcqmc_data['X_A'][:1000]
        # lcqmc_data['X_B'] = lcqmc_data['X_B'][:1000]
        # lcqmc_data['y'] = lcqmc_data['y'][:1000]
        lcqmc_data['y'] = [int(s) for s in lcqmc_data['y'][1:]]
        return lcqmc_data

    def run(self, params, batcher):
        lcqmc_embed = {'train': {}, 'test': {}}

        for key in self.lcqmc_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.lcqmc_data[key]['X_A'],
                                       self.lcqmc_data[key]['X_B'],
                                       self.lcqmc_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            text_data['A'] = [x for (x, y, z) in sorted_corpus]
            text_data['B'] = [y for (x, y, z) in sorted_corpus]
            text_data['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['A', 'B']:
                lcqmc_embed[key][txt_type] = []
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    lcqmc_embed[key][txt_type].append(embeddings)
                lcqmc_embed[key][txt_type] = np.vstack(lcqmc_embed[key][txt_type])
            lcqmc_embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = lcqmc_embed['train']['A']
        trainB = lcqmc_embed['train']['B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = lcqmc_embed['train']['y']

        # Test
        testA = lcqmc_embed['test']['A']
        testB = lcqmc_embed['test']['B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = lcqmc_embed['test']['y']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = KFoldClassifier(train={'X': trainF, 'y': trainY},
                              test={'X': testF, 'y': testY}, config=config)

        devacc, testacc, yhat = clf.run()
        testf1 = round(100*f1_score(testY, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for lcqmc.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainA), 'ntest': len(testA)}
