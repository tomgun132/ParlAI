import copy
from .build import build, make_path
from parlai.utils.misc import warn_once, str_to_msg
from parlai.core.teachers import ParlAIDialogTeacher


def _path(opt):
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    if datatype == 'test':
        warn_once("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    return make_path(opt, datatype + '.txt')


class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt)
        super().__init__(opt, shared)

    def _setup_data(self, path):
        print("[loading parlAI text data:" + path + "]")
        self.episodes = []
        self.num_exs = 0
        eps = []
        with open(path, encoding='utf-8') as read:
            for line in read:
                msg = str_to_msg(line.rstrip('\n'))
                if msg:
                    self.num_exs += 1
                    eps.append(msg)
                    if msg.get('episode_done', False):
                        self.episodes.append(eps)
                        eps = []
        if len(eps) > 0:
            # add last episode
            eps[-1]['episode_done'] = True
            self.episodes.append(eps)
