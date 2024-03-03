import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(256)
        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(512)
        self.enc_conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.enc_bn5 = nn.BatchNorm2d(512)

        self.enc_dropout = nn.Dropout(0.2)

        # Decoder
        self.dec_conv5 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_conv4 = nn.ConvTranspose2d(
            1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_conv3 = nn.ConvTranspose2d(
            512, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_conv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_conv1 = nn.ConvTranspose2d(
            256, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.final_conv = nn.Conv2d(6, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)))
        e2 = F.leaky_relu(self.enc_bn2(self.enc_conv2(e1)))
        e2 = self.enc_dropout(e2)
        e3 = F.leaky_relu(self.enc_bn3(self.enc_conv3(e2)))
        e3 = self.enc_dropout(e3)
        e4 = F.leaky_relu(self.enc_bn4(self.enc_conv4(e3)))
        e4 = self.enc_dropout(e4)
        e5 = F.leaky_relu(self.enc_bn5(self.enc_conv5(e4)))

        # Decoder
        d5 = F.leaky_relu(self.dec_conv5(e5))

        d5_cat = torch.cat([d5, e4], dim=1)
        d4 = F.leaky_relu(self.dec_conv4(d5_cat))

        d4_cat = torch.cat([d4, e3], dim=1)
        d3 = F.leaky_relu(self.dec_conv3(d4_cat))

        d3_cat = torch.cat([d3, e2], dim=1)
        d2 = F.leaky_relu(self.dec_conv2(d3_cat))

        d2_cat = torch.cat([d2, e1], dim=1)
        d1 = F.leaky_relu(self.dec_conv1(d2_cat))

        d1_cat = torch.cat([d1, x], dim=1)
        out = self.final_conv(d1_cat)

        return torch.sigmoid(out)
