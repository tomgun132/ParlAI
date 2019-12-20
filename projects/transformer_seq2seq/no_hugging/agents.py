from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.core.torch_generator_agent import TorchGeneratorAgent

from parlai.agents.transformer.modules import TransformerGeneratorModel
from parlai.agents.transformer.transformer import add_common_cmdline_args


class TransformerGeneratorAgent(TorchGeneratorAgent):
    """
    TransformerGeneratorAgent.

    Implementation of TorchGeneratorAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def load_state_dict(self, state_dict):
        """
        override to be able to load encoder from bi-encoder reddit model
        """

        try:
            """load from full generator model"""
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            """load encoder only from bi-encoder model"""
            # ? try loading cand_encoder into decoder probably
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                print(key)
                new_key = None
                if 'context' in key:
                    new_key = key.replace('context_', '')
            #     if 'cand_encoder' in key: # and is_copy_decoder:
            #         new_key = key.replace('cand_encoder', 'decoder').replace('attention', 'self_attention')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)

            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            self.model.load_state_dict(state_dict, strict=False)
