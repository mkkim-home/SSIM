
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb


class SSIM(nn.Module):
    def __init__(self, input_size, enc_hid_size, dec_hid_size, output_size):
        super(SSIM, self).__init__()

        self.input_size = input_size
        self.enc_hid_size = enc_hid_size  # size of hidden state at Encoder
        self.dec_hid_size = dec_hid_size  # size of hidden state at Decoder
        self.output_size = output_size

        self.enc_layer = 1
        self.dec_layer = 1

        self.dropout = 0.2

        self.encoder = nn.LSTM(self.input_size, self.enc_hid_size,
                               num_layers=self.enc_layer, batch_first=True,
                               dropout=self.dropout, bidirectional=True)  # 여기 hidden_size size확인
        self.attn = nn.Linear(self.dec_hid_size + self.enc_hid_size * 2,
                                   self.dec_hid_size)
        self.v = nn.Parameter(torch.rand(self.dec_hid_size))
        self.decoder = nn.LSTM(self.enc_hid_size * 2 + self.output_size, self.dec_hid_size,
                               num_layers=self.dec_layer, batch_first=True,
                               dropout=self.dropout)
        self.linear = nn.Linear(self.enc_hid_size * 2 + self.dec_hid_size,
                                self.output_size)

    def forward(self, input):
        batch_size = input.size(0)
        src_len = input.size(1)

        output = torch.zeros(batch_size, self.output_size)

        ''' encoder '''
        enc_h0 = torch.randn(self.enc_layer * 2, batch_size, self.enc_hid_size)  # 2 for bi-LSTM
        enc_c0 = torch.randn(self.enc_layer * 2, batch_size, self.enc_hid_size)

        encoder_outputs, (enc_h, enc_c) = self.encoder(input, (enc_h0, enc_c0))

        ''' decoder with attention '''
        dec_h0 = torch.randn(self.dec_layer, batch_size, self.dec_hid_size)
        dec_c0 = torch.randn(self.dec_layer, batch_size, self.dec_hid_size)

        y0 = torch.zeros(self.output_size).repeat(batch_size)  # decoder with attention을 어떻게 시작해야 하는지 모르겠다...
        # pdb.set_trace()

        hidden = dec_h0
        y = y0.unsqueeze(1)  # output dimension이 달라지면, 여기 unsqueeze 달라져야 할지도...
        for i in range(src_len):
            ''' attention '''
            hidden = torch.unbind(hidden, dim=0)[0]
            hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

            energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
            energy = energy.permute(0, 2, 1)

            v = self.v.repeat(batch_size, 1).unsqueeze(1)

            attention = torch.bmm(v, energy).squeeze(1)
            attention = F.softmax(attention, dim=1)

            attention = attention.unsqueeze(2).repeat(1, 1, self.enc_hid_size * 2)

            ''' context '''
            context = attention * encoder_outputs
            context = torch.sum(context, dim=1)

            ''' decoder lstm '''
            dec_input = torch.cat((y, context), dim=1)

            decoder_outputs, (dec_h, dec_c) = self.decoder(dec_input.unsqueeze(1), (dec_h0, dec_c0))

            ''' dense layer '''
            lin_input = torch.cat((decoder_outputs.squeeze(), context), dim=1)

            # pdb.set_trace()
            y = self.linear(lin_input)
            output = torch.cat((output, y), dim=1)

            hidden = decoder_outputs.permute(1, 0, 2)

        return output[:, 1:]
