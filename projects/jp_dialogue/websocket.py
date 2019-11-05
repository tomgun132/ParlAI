import tornado
from tornado.websocket import WebSocketHandler
from tornado.escape import json_decode, json_encode
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from ja_sentpiece_tokenizer import FullTokenizer

SHARED = {}
PORT = 8997

class ParlAIChatbot(WebSocketHandler):

    def open(self):
        print('connection open')

    def on_close(self):
        print('connection close')

    def on_message(self, message):
        try:
            json = json_decode(message)
            if json['speaker'] == 'user':
                text = json['text'].strip()
                print("input message: ", text)
                SHARED['conv_history']['input'].append(text)
                parsed_text = SHARED['tokenizer'].parse(text)
                reply = {'episode_done': False, 'text': parsed_text}
                if 'nlu' in SHARED:
                    parsed = SHARED['nlu'].parse(text)
                    topic = parsed['intent']['name']
                    reply['topic'] = topic
                SHARED['agent'].observe(reply)
                model_res = SHARED['agent'].act()
                resp_cands = model_res['text_candidates']
                # edited_resp_cands = self.block_repeat(resp_cands, self.history)
                resp = ''.join(resp_cands[0].replace('▁','').split())
                print("response: ", resp)
                output = {"response": resp, 'status': True}
                SHARED['conv_history']['response'].append(resp)
                SHARED['conv_history']['candidates'].append(resp_cands)
                self.write_message(json_encode(output))
        except Exception as e:
            self.write_message(json_encode({"response": str(e), "status": False}))
            print(e)


def setup_interactive(shared):
    """Build and parse CLI opts."""
    parser = setup_args()
    parser.add_argument('--port', type=int, default=PORT, help='Port to listen on.')
    parser.add_argument(
        '--hist-size',
        type=int,
        default=3,
        help='number of conversation history to remember'
    )
    parser.add_argument(
        '--with-topic',
        type=bool,
        default=False,
        help='number of conversation history to remember'
    )
    SHARED['opt'] = parser.parse_args(print_args=False)

    SHARED['opt']['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
    SHARED['opt']['model_file'] = '/installation/~/ParlAI/data/models/rachel/biber_poly_ranker'
    SHARED['opt']['model'] = 'projects.jp_dialogue.jp_retrieval.retrieval_agents:BertJPRanker'
    SHARED['opt']['context_model'] = 'poly'


    # Create model and assign it to the specified task
    agent = create_agent(SHARED.get('opt'), requireModelExists=True)
    SHARED['agent'] = agent
    SHARED['world'] = create_task(SHARED.get('opt'), SHARED['agent'])
    SHARED['tokenizer'] = FullTokenizer(SHARED['opt']['datapath'] + '/models/')
    if SHARED['opt']['with_topic']:
        if 'json' not in SHARED['opt']['fixed_candidates_path']:
            raise AttributeError("With topic option must use topical candidates file")
        from rasa.nlu.model import Interpreter
        SHARED['nlu'] = Interpreter.load(SHARED['opt']['datapath'] + '/rachel/topic_model')
    SHARED['conv_history'] = {'input': [], 'response': [], 'candidates': [], 'candidates_scores': []}
    # show args after loading model
    parser.opt = agent.opt
    parser.print_args()
    return agent.opt


def main():
    opt = setup_interactive(SHARED)
    print("Starting application...")
    application = tornado.web.Application([
        (r'/neural_chatbot', ParlAIChatbot),
    ])
    try:
        application.listen(opt['port'])
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        from datetime import datetime
        import csv

        tornado.ioloop.IOLoop.current().stop()
        print("Closing Server...")
        time = str(datetime.today()).split('.')[0]
        time = '_'.join(time.split())
        with open(SHARED['opt']['datapath'] + "/models/rachel/logs/logs_%s.csv" % time, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['input', 'response', 'candidates'])
            for i, inp in enumerate(SHARED['conv_history']['input']):
                # cands = ["{}({:.4f})".format(cand, score) for
                # cand, score in zip(SHARED['conv_history']['candidates'][i], SHARED['conv_history']['candidates_scores'][i])]
                cands = [cand for cand in SHARED['conv_history']['candidates'][i]]
                cands = ";".join(cands)
                resp = SHARED['conv_history']['response'][i]
                writer.writerow([inp, resp, cands])


if __name__ == '__main__':
    main()
