import torch.nn as nn


def get_encoder(observation_type, nr_inputs, cnn_channels, batch_norm=True):

    # 84 => 20 => 9 => 7
    if observation_type == '84x84':
        if batch_norm:
            enc = nn.Sequential(
                nn.Conv2d(nr_inputs, cnn_channels[0], 8, stride=4),
                nn.BatchNorm2d(cnn_channels[0]),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[0], cnn_channels[1], 4, stride=2),
                nn.BatchNorm2d(cnn_channels[1]),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[1], cnn_channels[2], 3, stride=1),
                nn.BatchNorm2d(cnn_channels[2]),
                nn.ReLU(),
                )
        else:
            enc = nn.Sequential(
                nn.Conv2d(nr_inputs, cnn_channels[0], 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[0], cnn_channels[1], 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[1], cnn_channels[2], 3, stride=1),
                nn.ReLU(),
                )

    elif observation_type == '16x16':
        if batch_norm:
            enc = nn.Sequential(
                nn.Conv2d(nr_inputs, cnn_channels[0], 4, 2, 1),
                nn.BatchNorm2d(cnn_channels[0]),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[0], cnn_channels[1], 4, 2, 1),
                nn.BatchNorm2d(cnn_channels[1]),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[1], cnn_channels[2], 4),
                nn.BatchNorm2d(cnn_channels[2]),
                nn.ReLU())
        else:
            enc = nn.Sequential(
                nn.Conv2d(nr_inputs, cnn_channels[0], 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[0], cnn_channels[1], 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[1], cnn_channels[2], 4),
                nn.ReLU())

    elif observation_type == 'fc':
        if batch_norm:
            enc = nn.Sequential(
                nn.Linear(nr_inputs, cnn_channels[0]),
                nn.BatchNorm1d(cnn_channels[0]),
                nn.ReLU(),
                nn.Linear(cnn_channels[0], cnn_channels[1]),
                nn.BatchNorm1d(cnn_channels[1]),
                nn.ReLU())
                # nn.Linear(cnn_channels[1], cnn_channels[2]),
                # nn.BatchNorm1d(cnn_channels[2]),
                # nn.ReLU())
        else:
            enc = nn.Sequential(
                nn.Linear(nr_inputs, cnn_channels[0]),
                nn.ReLU(),
                nn.Linear(cnn_channels[0], cnn_channels[1]),
                nn.ReLU())
                # nn.Linear(cnn_channels[1], cnn_channels[2]),
                # nn.ReLU())

    elif observation_type == '32x32':
        if batch_norm:
            # Out = floor((In + 2*Padding - kernel_size)/stride + 1)
            enc = nn.Sequential(
                # 32
                nn.Conv2d(nr_inputs, cnn_channels[0], 4, stride=2, padding=1),
                nn.BatchNorm2d(cnn_channels[0]),
                nn.ReLU(),
                # (32 + 2 - 4)/2 + 1 =  16
                nn.Conv2d(cnn_channels[0], cnn_channels[1], 4, 2, 1),
                nn.BatchNorm2d(cnn_channels[1]),
                nn.ReLU(),
                # (16 + 2 - 4)/2 + 1 =  8
                nn.Conv2d(cnn_channels[1], cnn_channels[2], 4, 2),
                nn.BatchNorm2d(cnn_channels[2]),
                nn.ReLU())
                # (8 - 4)/2 + 1 =  3
        else:
            enc = nn.Sequential(
                nn.Conv2d(nr_inputs, cnn_channels[0], 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[0], cnn_channels[1], 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[1], cnn_channels[2], 4, 2),
                nn.ReLU())

    else:
        raise NotImplementedError(
            "observation_type {} not implemented".format(observation_type))

    return enc


def get_cnn_output_dimension(observation_type, cnn_channels):
    if observation_type == '84x84':
        return [cnn_channels[2], 7, 7]
    elif observation_type == '16x16':
        return [cnn_channels[2], 1, 1]
    elif observation_type == 'fc':
        return [cnn_channels[1]]
    # elif observation_type == 'fc':
    #     return [cnn_channels[2]]
    elif observation_type == '32x32':
        return [cnn_channels[2], 3, 3]


def get_decoder(observation_type, nr_inputs, cnn_channels, batch_norm=True):
    dec_mean = nn.Sigmoid()
    dec_std = None
    if observation_type == '84x84':
        if batch_norm:
            decoder = nn.Sequential(
                # 32 x 7 x 7
                nn.ConvTranspose2d(cnn_channels[2], cnn_channels[1], kernel_size=3,
                                stride=1, padding=0),
                # L_out = (7-1)*stride - 2*padding + kernel_size = 6 + 3 = 9
                nn.BatchNorm2d(cnn_channels[1]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(cnn_channels[1], cnn_channels[0], 4, 2, 0),
                nn.BatchNorm2d(cnn_channels[0]),
                # L_out = (9-1)*stride - 2*padding + kernel_size = 8*2 + 4 = 20
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(cnn_channels[0], nr_inputs, 8, 4, 0),
                # L_out = (20-1)*stride - 2*padding + kernel_size = 19*2 + 8 = 76 + 8 = 84
            )
        else:
            decoder = nn.Sequential(
                nn.ConvTranspose2d(cnn_channels[2], cnn_channels[1], kernel_size=3,
                                stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(cnn_channels[1], cnn_channels[0], 4, 2, 0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(cnn_channels[0], nr_inputs, 8, 4, 0),
            )
    elif observation_type == '16x16':
        if batch_norm:
            decoder = nn.Sequential(
                nn.ConvTranspose2d(cnn_channels[2], cnn_channels[1], 4, 1, 0),
                nn.BatchNorm2d(cnn_channels[1]),
                nn.ReLU(),
                # 4 x 4
                nn.ConvTranspose2d(cnn_channels[1], cnn_channels[0], 4, 2, 1),
                nn.BatchNorm2d(cnn_channels[0]),
                nn.ReLU(),
                # 8 x 8
                nn.ConvTranspose2d(cnn_channels[0], nr_inputs, 4, 2, 1)
                # 16 x 16
            )
        else:
            decoder = nn.Sequential(
                nn.ConvTranspose2d(cnn_channels[2], cnn_channels[1], 4, 1, 0),
                nn.ReLU(),
                nn.ConvTranspose2d(cnn_channels[1], cnn_channels[0], 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(cnn_channels[0], nr_inputs, 4, 2, 1)
            )
    elif observation_type == 'fc':
        if batch_norm:
            decoder = nn.Sequential(
                # nn.Linear(cnn_channels[2], cnn_channels[1]),
                # nn.BatchNorm1d(cnn_channels[1]),
                # nn.ReLU(),
                nn.Linear(cnn_channels[1], cnn_channels[0]),
                nn.BatchNorm1d(cnn_channels[0]),
                nn.ReLU(),
            )
        else:
            decoder = nn.Sequential(
                # nn.Linear(cnn_channels[2], cnn_channels[1]),
                # nn.ReLU(),
                nn.Linear(cnn_channels[1], cnn_channels[0]),
                nn.ReLU(),
            )
        dec_mean = nn.Linear(cnn_channels[0], nr_inputs)
        dec_std = nn.Sequential(
            nn.Linear(cnn_channels[0], nr_inputs),
            nn.Softplus())
    elif observation_type == '32x32':
        if batch_norm:
            decoder = nn.Sequential(
                nn.ConvTranspose2d(cnn_channels[2], cnn_channels[1], 4, 2, 0),
                nn.BatchNorm2d(cnn_channels[1]),
                nn.ReLU(),
                # 8 x 8
                nn.ConvTranspose2d(cnn_channels[1], cnn_channels[0], 4, 2, 1),
                nn.BatchNorm2d(cnn_channels[0]),
                nn.ReLU(),
                # 16 x 16
                nn.ConvTranspose2d(cnn_channels[0], nr_inputs, 4, 2, 1)
                # 32 x 32
            )
        else:
            decoder = nn.Sequential(
                nn.ConvTranspose2d(cnn_channels[2], cnn_channels[1], 4, 2, 0),
                nn.ReLU(),
                nn.ConvTranspose2d(cnn_channels[1], cnn_channels[0], 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(cnn_channels[0], nr_inputs, 4, 2, 1)
            )

    return decoder, dec_mean, dec_std
