from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import io

from senteval.tools.validation import KFoldClassifier

from sklearn.metrics import f1_score


class PAWSXEval(object):
    def __init__(self, task_path, tokenizer, seed=1111):
        logging.info('***** Transfer task : PAWSX *****\n\n')
        self.seed = seed
        self.tokenizer = tokenizer
        train = self.loadFile(os.path.join(task_path,
                              'PAWSX.train.data'))
        test = self.loadFile(os.path.join(task_path,
                             'PAWSX.test.data'))
        self.pawsx_data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.pawsx_data['train']['X_A'] + \
                  self.pawsx_data['train']['X_B'] + \
                  self.pawsx_data['test']['X_A'] + self.pawsx_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        pawsx_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                if len(text) != 3:
                    continue
                pawsx_data['X_A'].append(self.tokenizer.tokenize(text[0]))
                pawsx_data['X_B'].append(self.tokenizer.tokenize(text[1]))
                pawsx_data['y'].append(text[2])
        # pawsx_data['X_A'] = pawsx_data['X_A'][:1000]
        # pawsx_data['X_B'] = pawsx_data['X_B'][:1000]
        # pawsx_data['y'] = pawsx_data['y'][:1000]
        pawsx_data['y'] = [int(s) for s in pawsx_data['y'][1:]]
        return pawsx_data

    def run(self, params, batcher):
        pawsx_embed = {'train': {}, 'test': {}}

        for key in self.pawsx_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.pawsx_data[key]['X_A'],
                                       self.pawsx_data[key]['X_B'],
                                       self.pawsx_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            text_data['A'] = [x for (x, y, z) in sorted_corpus]
            text_data['B'] = [y for (x, y, z) in sorted_corpus]
            text_data['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['A', 'B']:
                pawsx_embed[key][txt_type] = []
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    pawsx_embed[key][txt_type].append(embeddings)
                pawsx_embed[key][txt_type] = np.vstack(pawsx_embed[key][txt_type])
            pawsx_embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = pawsx_embed['train']['A']
        trainB = pawsx_embed['train']['B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = pawsx_embed['train']['y']

        # Test
        testA = pawsx_embed['test']['A']
        testB = pawsx_embed['test']['B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = pawsx_embed['test']['y']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = KFoldClassifier(train={'X': trainF, 'y': trainY},
                              test={'X': testF, 'y': testY}, config=config)

        devacc, testacc, yhat = clf.run()
        testf1 = round(100*f1_score(testY, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for pawsx.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainA), 'ntest': len(testA)}
