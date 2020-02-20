

import copy
import parlai.core.agents as core_agents
from parlai.core.agents import create_task_agent_from_taskname
from parlai.core.teachers import FixedDialogTeacher
from .build import build

import json
import os
import random


TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
START_ENTRY = {'message': '__SILENCE__', 'agent': 'agent_2', 'sentiment': 'Neutral', 'knowledge_source': [], 'turn_rating': ''}

def _first_val(dictionary):
    vals = list(dictionary.values())
    if len(vals) > 0:
        return vals[0]
    return ''


def _first_key(dictionary):
    keys = list(dictionary.keys())
    if len(keys) > 0:
        return keys[0]
    return ''

def _path(opt, split='freq'):
    build(opt)
    dp = os.path.join(opt['datapath'], 'topical_chat')
    dt = opt.get('datatype', 'train').split(':')[0]
    if dt == 'train':
        df = 'train.json'
    else:
        df = '{}_{}.json'.format(dt, split)
    return os.path.join(dp, df)

def _knowledge_path(opt, split='freq'):
    dp = os.path.join(opt['datapath'], 'topical_chat', 'reading')
    dt = opt.get('datatype', 'train').split(':')[0]
    if dt == 'train':
        df = 'train.json'
    else:
        df = '{}_{}.json'.format(dt, split)
    return os.path.join(dp, df)

class BaseTopicalTeacher(FixedDialogTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Wizard Dialog Knowledge arguments')
        agent.add_argument(
            '--valid-test-type',
            type=str,
            choices=['freq', 'req'],
            default='freq',
            help='Choose which valid or test data to use '
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt

        split = opt.get('valid_test_type', 'freq')

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.data_path = _path(opt, split=split)
            self._setup_data()
        self.num_exs = sum(len(d['content']) for d in self.data)
        self.num_eps = len(self.data)
        self.reset()

    def _setup_data(self):
        print('loading: ' + self.data_path)
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            self.data = []
            for k,v in data.items():
                v['id'] = k
                self.data.append(v)

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        input_turn = d['content'][entry_idx]
        episode_done = entry_idx == (self.len_episode(episode_idx) - 1)

        action = {
            'id': d['id'],
            'text': input_turn['message'],
            'sentiment': input_turn['sentiment'],
            'knowledge_source': input_turn['knowledge_source'],
            'config': d['config'],
            'episode_done': episode_done,
        }

        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared

class TopicalDialogTeacher(BaseTopicalTeacher):

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        split = opt.get('valid_test_type', 'freq')
        knowledge_path = _knowledge_path(opt, split=split)
        with open(knowledge_path, 'r') as f:
            self.knowledge = json.load(f)

    def get(self, episode_idx, entry_idx=0):
        # Sometimes we're speaker 1 and sometimes we're speaker 2
        speaker_id = episode_idx % 2
        d = self.data[episode_idx // 2]

        entries = [START_ENTRY] + d['content']
        input_turn = entries[speaker_id + 2 * entry_idx]
        label_turn = entries[1 + speaker_id + 2 * entry_idx]

        episode_done = 2 * entry_idx + speaker_id + 1 >= len(d['content']) - 1
        if input_turn['message'] == '__SILENCE__':
            input_kn = None
        else:
            input_kn = self.knowledge[d['id']][input_turn['agent']]
        label_kn = self.knowledge[d['id']][label_turn['agent']]
        article = self.knowledge[d['id']]['article']
        action = {
            'text': input_turn['message'],
            'sentiment': input_turn['sentiment'],
            'knowledge_source': input_turn['knowledge_source'],
            'input_knowledges': input_kn,
            'output_knowledges': label_kn,
            'labels': [label_turn['message']],
            'config': d['config'],
            'article': article,
            'episode_done': episode_done,
        }

        return action

class TopicalSelectionTeacher(BaseTopicalTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        split = opt.get('valid_test_type', 'freq')
        knowledge_path = _knowledge_path(opt, split=split)
        with open(knowledge_path, 'r') as f:
            self.knowledge = json.load(f)

    def get(self, episode_idx, entry_idx=0):
        action = super().get()
        


class DefaultTeacher(TopicalDialogTeacher):
    pass
