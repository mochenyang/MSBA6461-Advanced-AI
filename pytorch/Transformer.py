import numpy as np
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.tensorboard import SummaryWriter

# Simulate data
# set random seed for reproducibility
np.random.seed(123)
X = np.random.uniform(size = (5000, 10))
Y = np.mean(X, axis = 1)
X_train = X[:4000]
X_test = X[4000:]
Y_train = Y[:4000]
Y_test = Y[4000:]
#print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# vocab has single-digits, space, start, end
VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', 's', 'e']
# for simplicity, we restrict each input/output number to 8 digits
MAX_DIGITS = 8

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, vocab):
        self.X = X
        self.Y = Y
        self.vocab = vocab
        # the "index" method is defined below
        self.X_indexed = self.index(X, 'X')
        self.Y_indexed = self.index(Y, 'Y')

    # The "index" method converts either an input vector or an output value to a sequence of token indices
    def index(self, data, type):
        data_indexed = []
        for row in data:
            if type == 'Y':
                # in this case, row is a scalar, we convert it to a string and remove the "0." prefix
                # the '{:8f}'.format(...) part ensures the number has 8 digits after the decimal point, and converts it to a string
                # the '[2:]' part removes the "0." prefix
                row_str = '{:.8f}'.format(row)[2:]
            if type == 'X':
                # in this case, we do the same processing to each feature value, then concatenate them to a longer sequence, separated by blank spaces
                row_str = ' '.join(['{:.8f}'.format(x)[2:] for x in row])
            # also need to prepend 's' and append 'e' to the sequence
            row_str = 's' + row_str + 'e'
            # convert to indices in vocabulary
            row_idx = [self.vocab.index(c) for c in row_str]
            data_indexed.append(row_idx)
        return np.array(data_indexed)

    def __len__(self):
        # this is a required method in custom dataset classes, it should return size of data (i.e., number of rows)
        return len(self.X_indexed)

    def __getitem__(self, idx):
        # this is also a required method, it should return the item at the given index
        src = torch.tensor(self.X_indexed[idx], dtype=torch.long)
        tgt = torch.tensor(self.Y_indexed[idx], dtype=torch.long)
        return src, tgt
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        :param vocab_size: the size of the vocabulary
        :param d_model: the embedding dimension
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        """
        :param tokens: the input tensor with shape (batch_size, seq_len)
        :return: the tensor after token embedding with shape (batch_size, seq_len, d_model)
        """
        return self.embedding(tokens)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        """
        :param d_model: the embedding dimension
        :param max_len: the maximum length of the sentence
        """
        super(PositionalEncoding, self).__init__()
        # setting max_len to 100 here, because the largest input sequence is 91 tokens long (10 * 8 digits + 9 spaces + 1 start + 1 end), so 100 is enough
        # intialize the positional encoding, pe.shape = (max_len, d_model)        
        pe = torch.zeros(max_len, d_model)
        # generate a tensor of shape (max_len, 1), with values from 0 to max_len - 1, to represent all unique positions
        # the unsqueeze(1) operation adds a dimension after the first dimension, so the shape changes from (max_len,) to (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # calculate scaling factors for each dimension of the positional encoding, see the formula in the first section of this notebook
        scaling_factors = torch.tensor([10000.0 ** (-2 * i / d_model) for i in range(d_model // 2)])
        # now populate the positional encoding tensor with values, even indices use sine functions, odd indices use cosine functions
        pe[:, 0::2] = torch.sin(position * scaling_factors)  # pe[:, 0::2].shape = (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * scaling_factors)  # pe[:, 1::2].shape = (max_len, d_model/2)
        # add a batch dimension to the positional encoding tensor so that it's compatible with the input tensor. pe.shape = (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # register the positional encoding tensor as a buffer, so that it will be stored as part of the model's "states" and won't be updated during training
        # this is desirable because we don't want the positional encoding to be trained, we want it to be fixed
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: the input tensor with shape (batch_size, seq_len, d_model)
        :return: the tensor after adding positional encoding with shape (batch_size, seq_len, d_model)
        """
        # for a given input tensor x, add the positional encoding to it
        # x.size(1) gets the second dimensions of x, which is dimension that contains the element indices in the sequence
        x = x + self.pe[:, :x.size(1)]
        return x
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, vocab_size):
        """
        :param d_model: the embedding dimension
        :param nhead: the number of heads in multi-head attention
        :param num_encoder_layers: the number of blocks in the encoder
        :param num_decoder_layers: the number of blocks in the decoder
        :param dim_feedforward: the dimension of the feedforward network
        """
        super(Seq2SeqTransformer, self).__init__()
        # note that, in many other tasks (e.g., translation), you need two different token embeddings for the source and target languages
        # here, however, because both input and output use the same vocabulary, we can use the same token embedding for both
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        # the transformer model is constructed with the Transformer module, which takes care of all the details
        # the batch_first=True argument means the input and output tensors are of shape (batch_size, seq_len, d_model)
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first=True)
        # the generator is a simple linear layer that projects the transformer output to the vocabulary size
        # it generates the logits for each token in the vocabulary, will be used for computing loss and making predictions
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """
        :param src: the sequence to the encoder (required). with shape (batch_size, seq_len, d_model)
        :param tgt: the sequence to the decoder (required). with shape (batch_size, seq_len, d_model)
        :param src_mask: the additive mask for the src sequence (optional). with shape (batch_size, seq_len, seq_len)
        :param tgt_mask: the additive mask for the tgt sequence (optional). with shape (batch_size, seq_len, seq_len)
        :param src_padding_mask: the additive mask for the src sequence (optional). with shape (batch_size, 1, seq_len)
        :param tgt_padding_mask: the additive mask for the tgt sequence (optional). with shape (batch_size, 1, seq_len)
        :param memory_key_padding_mask: the additive mask for the encoder output (optional). with shape (batch_size, 1, seq_len)
        :return: the decoder output tensor with shape (batch_size, seq_len, d_model)
        """
        # separately embed the source and target sequences
        src_emb = self.positional_encoding(self.tok_emb(src))
        tgt_emb = self.positional_encoding(self.tok_emb(tgt))
        # Important: we don't need any masks for source sequence, or any padding masks, nor do we need a mask for decoder attending to the encoder
        # but we do need a mask for the target sequence -- this is a "causal mask", which prevents the decoder from attending to subsequent tokens during training
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1))
        outs = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.generator(outs)
    
    # The transformer also have an encode method and a decode method
    # the encode method takes the source sequence and produce the context vector (which pytorch calls "memory")
    # the decoder method takes the target sequence and the context vector, and produce the output sequence
    def encode(self, src):
        """
        :param src: the sequence to the encoder (required). with shape (batch_size, seq_len, d_model)
        :return: the encoder output tensor with shape (batch_size, seq_len, d_model)
        """
        return self.transformer.encoder(self.positional_encoding(self.tok_emb(src)))
    
    def decode(self, tgt, memory):
        """
        :param tgt: the sequence to the decoder (required). with shape (batch_size, seq_len, d_model)
        :param memory: the sequence from the last layer of the encoder (required). with shape (batch_size, seq_len, d_model)
        :return: the decoder output tensor with shape (batch_size, seq_len, d_model)
        """
        return self.transformer.decoder(self.positional_encoding(self.tok_emb(tgt)), memory)
    

if __name__ == "__main__":
    train_dataset = CustomDataset(X_train, Y_train, VOCAB)
    test_dataset = CustomDataset(X_test, Y_test, VOCAB)
    
    # specify model parameters and training parameters
    VOCAB_SIZE = len(VOCAB)
    EMB_SIZE = 256
    NHEAD = 4
    FFN_HID_DIM = 128
    BATCH_SIZE = 32
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    NUM_EPOCHS = 1000
    
    # instantiate the model
    model = Seq2SeqTransformer(EMB_SIZE, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FFN_HID_DIM, VOCAB_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create DataLoader for batching
    # for eval_loader, we load data one at a time for better demonstration of what happens -- in practice you can also batch it
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize TensorBoard writer and loss storage
    # make sure to create a "logs" folder in the parent directory before running the code
    writer = SummaryWriter('../logs/transformer')
    
    # training
    step = 0
    for epoch in range(NUM_EPOCHS):
        # start model training
        model.train()
        # initialize total loss for the epoch
        total_loss = 0
        for src, tgt in train_loader:
            optimizer.zero_grad()        
            # Separate the input and target sequences for teacher forcing
            # tgt_input has everything except the last token
            # tgt_output has everything except the first token
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            # Forward pass with teacher forcing, logits has shape (batch_size, seq_len, vocab_size)
            logits = model(src, tgt_input)
            # Calculate loss. The .reshape(-1) flattens the logits to (batch_size * seq_len, vocab_size)
            outputs = logits.reshape(-1, logits.shape[-1])
            # also flatten the ground truth outputs to shape (batch_size * seq_len)
            tgt_out = tgt_output.reshape(-1)
            loss = criterion(outputs, tgt_out)
            loss_item = loss.item()
            total_loss += loss_item
            writer.add_scalar('Loss/train_step', loss_item, step)
            loss.backward()
            optimizer.step()
            step += 1
        writer.add_scalar('Loss/train_epoch', total_loss, epoch)
        print(f"Epoch: {epoch}, Training Loss: {total_loss}")        
        
        # monitor loss on test set
        model.eval()
        test_loss = 0
        test_step = epoch * len(eval_loader)      
        with torch.no_grad():
            for test_batch_idx, (src, tgt) in enumerate(eval_loader):
                encoder_output = model.encode(src)
                # decoding starts with the "start" token
                tgt_idx = [VOCAB.index('s')]
                pred_num = '0.'
                for i in range(MAX_DIGITS + 1): # 0 to 8 (9 iterations for 8 digits, including end token)
                    # prepare the input tensor for the decoder, adding the batch dimension
                    decoder_input = torch.LongTensor(tgt_idx).unsqueeze(0)
                    # the decoder output has shape (1, seq_len, d_model) and the last position in sequence is the prediction for next token
                    decoder_output = model.decode(decoder_input, encoder_output)
                    # the predicted logits has shape (1, seq_len, vocab_size)
                    logits = model.generator(decoder_output)
                    # calculate test loss based on most recent token prediction, that is logits[:, -1, :]
                    step_loss = criterion(logits[:, -1, :], tgt[0][i+1].unsqueeze(0)).item()
                    test_loss += step_loss
                    writer.add_scalar('Loss/test_step', step_loss, test_step)
                    test_step += 1
                    # the actual predicted token is the one with highest logit score
                    # here, .argmax(2) makes sure the max is taken on the last dimension, which is the vocabulary dimension, and [:, -1] makes sure that we are looking at the last position in the sequence
                    pred_token = logits.argmax(2)[:,-1].item()
                    # append the predicted token to target sequence as you go
                    tgt_idx.append(pred_token)
                    pred_num += VOCAB[pred_token]
                    if pred_token == VOCAB.index('e'):
                        break
                # Convert the predicted sequence to a number - if you want, you can use it to compute other metrics such as RMSE
                try:
                    pred_num = float(pred_num)  # Convert the accumulated string to a float
                except ValueError:
                    pred_num = 0.0  # Handle any conversion errors gracefully
        writer.add_scalar('Loss/test_epoch', test_loss, epoch)
        print("Test Loss: ", test_loss)
    
    # Close TensorBoard writer
    writer.close()