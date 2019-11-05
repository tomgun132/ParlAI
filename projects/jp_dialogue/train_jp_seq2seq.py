from parlai.scripts.train_model import TrainLoop
from parlai.scripts.train_model import setup_args as tm_setupargs

def setup_args():
    """Defaults for baseline model"""
    parser = tm_setupargs()

    parser.set_defaults(
        task='projects.jp_dialogue.tasks.agents',
        model='projects.jp_dialogue.jp_seq2seq.jp_seq2seq:ControllableJPSeq2seq',  # noqa: E501
        dict_lower=True,
        dict_include_valid=True,
        # dict_maxexs=60000,
        dict_minfreq=10,
        datatype='train',
        batchsize=64,
        hiddensize=1024,
        embeddingsize=300,
        attention='general',
        numlayers=2,
        rnn_class='lstm',
        learningrate=3,
        dropout=0.1,
        gradient_clip=0.1,
        lookuptable='enc_dec',
        optimizer='sgd',
        embedding_type='fasttext',
        momentum=0.9,
        bidirectional=False,
        context_length=-1,
        person_tokens=True,
        add_p1_after_newln=True,
        beam_min_n_best=30,
        validation_every_n_secs=300,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=12,
        log_every_n_secs=10,
        dict_tokenizer='split',
        load_from_checkpoint=False,
        save_after_valid = True,
        tensorboard_log=True,
        tensorboard_tag='task,batchsize,hiddensize,embeddingsize,attention,numlayers,rnn_class,learningrate,dropout,gradient_clip',
        tensorboard_metrics='ppl,loss,accuracy',
        train_folder='jp_dialogue'
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    print('-------------------------training')
    TrainLoop(opt).train()
    # python train_jp_seq2seq.py -mf \installation\~\ParlAI\data\models\jp_dialogue\jp_twitter_base
    # --dict-class projects.jp_dialogue.jp_retrieval.retrieval_agents:UniBiDictionaryAgent

    # finetune:
    # python train_jp_seq2seq.py -mf \installation\~\ParlAI\data\models\rachel\rachel
    # --init-model models:jp_dialogue/jp_twitter_base_bpe --dict_file models:jp_dialogue/jp_twitter_base_bpe
    # -bs 16 --dict-tokenizer bpe --train-folder rachel
