#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from natto import MeCab
from ja_sentpiece_tokenizer import FullTokenizer
import random
import os
import time
characters = ["私の夢は宇宙に行くことです。",
"私の目標は全世界の人に会うことです。",
"私の目標は全ての国で働くことです。",
"私は勉強が好きです。",
"私はSpaceXが好きです。",
"私はElon Muskを尊敬しています。",
"私はNASAに行きたいです。",
"私は仕事が好きです。",
"私はいつも頑張ります。"]

def block_repeat(cands, history):
    new_cands = []
    for cand in cands:
        if cand not in history:
            new_cands.append(cand)
    return new_cands

def interactive(opt):
    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, [agent])
    if 'bert' in opt['model_file']:
        tokenizer = FullTokenizer(opt['datapath'] + '/models/')
    else:
        tokenizer = MeCab('-Owakati')
    not_quit = True
    i = 0
    chosen = random.sample(characters, k=4)
    # persona_line = ''
    # for persona in chosen:
    #     persona_line += 'your persona: {}\\n'.format(tokenizer.parse(persona))
    history = []
    while not_quit:
        text = input('Enter your message:\n')
        # if i == 0:
        #     text = persona_line + text
        #     print(text)
        if text == 'quit':
            return "また会いましょう。"
        
        start = time.time()
        if len(history) > opt.get('hist_size'):
            del history[0]
        text = tokenizer.parse(text)
        print(text)
        obs = {'text': text, 'episode_done': False}
        agent.observe(obs)
        out = agent.act()
        if 'bert' in opt['model_file']:
            resp_cands = out['text_candidates']
            print('Candidates: \n')
            for i, cand in enumerate(resp_cands):
                line = "{}. {}({})\n".format(i, cand, out['scores'][i])
                print(line)
            resp_cands = block_repeat(resp_cands, history)
            resp = resp_cands[0]
            history.append(resp)
        else:
            resp = out['text']
        resp = ''.join(resp.replace('__unk__', '々').replace('▁','').split())
        print('response: %s' % resp)
        end = time.time()
        print('process time: {}'.format(end - start))

if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    parser.add_argument(
        '--hist-size',
        type=int,
        default=3,
        help='number of conversation history to remember'
    )
    parser.set_params(batchsize=1, beam_size=20, beam_min_n_best=10)

    # print('\n' + '*' * 80)
    # print('WARNING: This dialogue model is a research project that was trained on a')
    # print(
    #     'large amount of open-domain Twitter data. It may generate offensive content.'
    # )
    # print('*' * 80 + '\n')

    last = interactive(parser.parse_args(print_args=False))
    print(last)
    # >python interactive.py -mf \installation\~\ParlAI\data\models\rachel\controllable_rachel
    # --weighted-decoding extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_2gram:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,lastuttsim:5

    # Interactive for Bert ranker
    # python interactive.py -mf /installation/~/ParlAI/data/models/rachel/bibert_ranker
    # -m projects.jp_dialogue.jp_retrieval.retrieval_agents:BertJPRanker --encode-candidate-vecs true --eval-candidates fixed
    # --fixed-candidates-path  /installation/~/ParlAI/data/rachel/tokenized_cands.txt --rank-top-k 10
