import torch

class Config(object):
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        
        self.max_length = 30
        self.reverse = False

        self.input_lang_n_words = None
        self.output_lang_n_words = None

        self.teacher_forcing_ratio = 0.5

        # Train parameters
        self.n_iters = 30000
        self.print_every = 1000
        self.plot_every=100
        self.learning_rate=0.005

        # Encoder architecture
        self.encoder_hidden_size = 256
        self.encoder_n_layers = 1

        # Decoder architecture
        self.decoder_hidden_size = 256
        self.decoder_output_size = 256
        self.decoder_n_layers = 1
        self.decoder_dropout_p = 0.1

        self.evaluate_n = 10