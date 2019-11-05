from parlai.scripts.train_model import TrainLoop
from parlai.scripts.train_model import setup_args as tm_setupargs

def setup_args():
    """Defaults for baseline model"""
    parser = tm_setupargs()
    parser.add_argument(
        '--train-folder',
        type=str,
        help='training data location'
    )
    parser.set_defaults(
        task='projects.jp_dialogue.tasks.agents',
        model='projects.jp_dialogue.jp_retrieval.retrieval_agents:PolyAIJPRanker',  # noqa: E501
        # dict_lower=True,
        # dict_include_valid=True,
        # dict_maxexs=60000,
        # dict_minfreq=5,
        datatype='train',
        batchsize=16,
        embeddingsize=320,
        attention='general',
        # numlayers=2,
        # rnn_class='lstm',
        learningrate=0.03,
        dropout=0.1,
        gradient_clip=0.1,
        optimizer='sgd',
        embedding_type='fasttext',
        momentum=0.9,
        bidirectional=False,
        context_length=-1,
        person_tokens=True,
        add_p1_after_newln=True,
        beam_min_n_best=30,
        validation_every_n_secs=300,
        validation_metric='accuracy',
        validation_metric_mode='max',
        validation_patience=12,
        log_every_n_secs=10,
        load_from_checkpoint=False,
        save_after_valid = True,
        tensorboard_log=True,
        tensorboard_tag='task,batchsize,hiddensize,embeddingsize,attention,numlayers,rnn_class,learningrate,dropout,gradient_clip',
        tensorboard_metrics='ppl,loss,accuracy',
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    TrainLoop(opt).train()

    # Train ranker base using Poly-Encoder
    #  python train_jp_ranker.py -mf /home/ubuntu/workspace/ParlAI/data/models/jp_dialogue/jp_twitter_ranker_base
    #  --model transformer/polyencoder -bs 128 --eval-batchsize 10 --dict-tokenizer bpe --warmup-updates 100
    #  --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 -lr 2e-04 --label-truncate 72 --text-truncate 360
    # --validation-metric accuracy --validation-metric-mode max --candidates batch --optimizer adamax --output-scaling 0.06
    # --variant xlm --reduction-type mean --share-encoders False --learn-positional-embeddings True --n-layers 12 --n-heads 12
    # --ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 --embedding-size 768 --activation gelu
    # --embeddings-scale False --n-segments 2 --learn-embeddings True --share-word-embeddings False --dict-endtoken __start__
    # --encode-candidate-vecs False

    # Train ranker base using PolyAIEncoder
    # python train_jp_ranker.py -mf /home/ubuntu/workspace/ParlAI/data/models/jp_dialogue/jp_ranker_base
    # --dict-class projects.jp_dialogue.jp_retrieval.retrieval_agents:UniBiDictionaryAgent
    # --lr-scheduler-decay 0.3 -lr 0.03 --candidates batch --eval-candidates batch --n-layers 1 --n-heads 5 -bs 512
    # --scoring-func dot --encode-candidate-vecs False --eval-batchsize 10 --embeddings-scale True
    # --variant xlm --learn-positional-embeddings True --ffn-size 2048 --attention-dropout 0.1
    # --relu-dropout 0.0 --dropout 0.1 --embedding-size 320 --activation gelu

    # Fine-tune bi-encoder model using bi-encoder pretrained from bert
    #  python train_jp_ranker.py -mf /home/ubuntu/workspace/ParlAI/data/models/rachel/bibert_ranker_nopersona
    # -m bert_ranker/bi_encoder_ranker -bs 16 --eval-batchsize 16 --train-folder rachel --warmup-updates 100 
    #--lr-scheduler-patience 0 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 --validation-metric accuracy 
    # --validation-metric-mode max --optimizer adamw --bert-aggregation mean --type-optimization top4_layers 
    # --eval-candidates batch --validation-every-n-secs -1 --num-epochs 100 --history-size 3


    # or from ParlAI main folder

    # python examples/train_model.py -mf /home/ubuntu/workspace/ParlAI/data/models/jp_dialogue/jp_finetuned_bibert_ranker
    # -t projects.jp_dialogue.tasks.agents -m bert_ranker/bi_encoder_ranker -bs 32 --eval-batchsize 10
    # --warmup-updates 100 --lr-scheduler-patience 0 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4
    # --validation-metric accuracy --validation-metric-mode max --optimizer adamw --bert-aggregation mean

    # Fine-tune polyAI model using twitter pretrained model
    # python train_jp_ranker.py -mf /home/ubuntu/workspace/ParlAI/data/models/rachel/polyai_ranker
    # --init-model /home/ubuntu/workspace/ParlAI/data/models/jp_dialogue/jp_ranker_base
    # --dict-file /home/ubuntu/workspace/ParlAI/data/models/jp_dialogue/jp_ranker_base.dict
    # --dict-class projects.jp_dialogue.jp_retrieval.retrieval_agents:UniBiDictionaryAgent --lr-scheduler-decay 0.3
    # -lr 0.03 --candidates batch --n-layers 1 --n-heads 5 -bs 128 --scoring-func dot --encode-candidate-vecs False
    # --eval-batchsize 10 --embeddings-scale True --variant xlm --learn-positional-embeddings True --ffn-size 2048
    # --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --embedding-size 320 --activation gelu
    # --validation-every-n-secs -1 --max_train_time 6900 --train-folder rachel

    # python train_jp_ranker.py -mf /home/ubuntu/workspace/ParlAI/data/models/rachel/bibert_poly_ranker_v1.1 
    # -m projects.jp_dialogue.jp_retrieval.retrieval_agents:BertJPRanker -bs 16 --eval-batchsize 16 
    # --train-folder rachel --warmup-updates 100 --lr-scheduler-patience 0 --lr-scheduler-patience 0 
    # --lr-scheduler-decay 0.4 --validation-metric accuracy --validation-metric-mode max 
    # --optimizer adamw --bert-aggregation mean --type-optimization top4_layers --validation-every-n-secs -1 
    # --num-epochs 100.0 --context-model poly --person-tokens False --history-size 3

