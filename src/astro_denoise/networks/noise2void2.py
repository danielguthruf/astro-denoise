import torch
from torch import nn

import astro_denoise.networks.basicblock as B


class UNet(nn.Module):
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
        bias=True,
    ):
        super(UNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode="C")

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = B.downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = B.downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = B.downsample_strideconv
        elif downsample_mode == "maxblurpool":
            downsample_block = B.downsample_maxblurpool
        else:
            raise NotImplementedError(f"downsample mode [{downsample_mode:s}] is not found")

        self.m_down1 = B.sequential(
            *[B.ResBlock(nc[0], nc[0], bias=bias, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=bias, mode="2"),
        )
        self.m_down2 = B.sequential(
            *[B.ResBlock(nc[1], nc[1], bias=bias, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=bias, mode="2"),
        )
        self.m_down3 = B.sequential(
            *[B.ResBlock(nc[2], nc[2], bias=bias, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=bias, mode="2"),
        )

        self.m_body = B.sequential(
            *[B.ResBlock(nc[3], nc[3], bias=bias, mode="C" + act_mode + "C") for _ in range(nb)]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = B.upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(f"upsample mode [{upsample_mode:s}] is not found")

        self.m_up3 = B.sequential(
            upsample_block(nc[3], nc[2], bias=bias, mode="2"),
            *[B.ResBlock(nc[2], nc[2], bias=bias, mode="C" + act_mode + "C") for _ in range(nb)],
        )
        self.m_up2 = B.sequential(
            upsample_block(nc[2], nc[1], bias=bias, mode="2"),
            *[B.ResBlock(nc[1], nc[1], bias=bias, mode="C" + act_mode + "C") for _ in range(nb)],
        )
        # self.m_up3 = B.sequential(*[B.ResBlock(nc[3], nc[2], bias=bias, mode='C'+ act_mode+'C') for _ in range(nb)])
        # self.m_up2 = B.sequential(*[B.ResBlock(nc[2], nc[1], bias=bias, mode='C'+ act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(
            upsample_block(nc[1], nc[0], bias=bias, mode="2"),
            *[B.ResBlock(nc[0], nc[0], bias=bias, mode="C" + act_mode + "C") for _ in range(nb)],
        )

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode="C")

    def forward(self, x0, pad=64, test=True):
        if test:
            x0 = torch.nn.functional.pad(x0, (pad, pad, pad, pad), "replicate")
        #        h, w = x.size()[-2:]
        #        paddingBottom = int(np.ceil(h/8)*8-h)
        #        paddingRight = int(np.ceil(w/8)*8-w)
        #        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        # x = self.m_up3(x)
        # x = self.m_up2(x)

        x = self.m_up1(x + x2)
        # x = self.m_tail(x+x1)
        x = self.m_tail(x)

        #        x = x[..., :h, :w]
        if test:
            x = x[:, :, pad:-pad, pad:-pad]
        return x


def generate_mask_base(input, ratio=0.9, size_window=(5, 5)):
    input = input.permute(1, 2, 0)
    size_data = input.shape
    num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

    mask = torch.ones(size_data).to("cuda")
    output = input

    for ich in range(size_data[2]):
        idy_msk = torch.randint(0, size_data[0], (num_sample,), dtype=torch.long)
        idx_msk = torch.randint(0, size_data[1], (num_sample,), dtype=torch.long)

        idy_neigh = torch.randint(
            -size_window[0] // 2 + size_window[0] % 2,
            size_window[0] // 2 + size_window[0] % 2,
            (num_sample,),
            dtype=torch.long,
        )
        idx_neigh = torch.randint(
            -size_window[1] // 2 + size_window[1] % 2,
            size_window[1] // 2 + size_window[1] % 2,
            (num_sample,),
            dtype=torch.long,
        )

        idy_msk_neigh = idy_msk + idy_neigh
        idx_msk_neigh = idx_msk + idx_neigh

        idy_msk_neigh = (
            idy_msk_neigh
            + (idy_msk_neigh < 0) * size_data[0]
            - (idy_msk_neigh >= size_data[0]) * size_data[0]
        )
        idx_msk_neigh = (
            idx_msk_neigh
            + (idx_msk_neigh < 0) * size_data[1]
            - (idx_msk_neigh >= size_data[1]) * size_data[1]
        )

        id_msk = (idy_msk, idx_msk, ich)
        id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)
        output[id_msk] = input[id_msk_neigh]
        mask[id_msk] = 0.0

    output = output.permute(2, 0, 1)
    mask = mask.permute(2, 0, 1)

    return output, mask


def generate_mask_median(input, ratio=0.9, size_window=(5, 5), s=5):
    input = input.permute(1, 2, 0)
    size_data = input.shape
    num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

    mask = torch.ones(size_data).to("cuda")
    output = input

    for ich in range(size_data[2]):
        idy_msk = torch.randint(0, size_data[0], (num_sample,), dtype=torch.long)
        idx_msk = torch.randint(0, size_data[1], (num_sample,), dtype=torch.long)
        id_msk = (idy_msk, idx_msk, ich)

        input_list = []
        for i in range(s):
            idy_neigh = torch.randint(
                -size_window[0] // 2 + size_window[0] % 2,
                size_window[0] // 2 + size_window[0] % 2,
                (num_sample,),
                dtype=torch.long,
            )
            idx_neigh = torch.randint(
                -size_window[1] // 2 + size_window[1] % 2,
                size_window[1] // 2 + size_window[1] % 2,
                (num_sample,),
                dtype=torch.long,
            )

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = (
                idy_msk_neigh
                + (idy_msk_neigh < 0) * size_data[0]
                - (idy_msk_neigh >= size_data[0]) * size_data[0]
            )
            idx_msk_neigh = (
                idx_msk_neigh
                + (idx_msk_neigh < 0) * size_data[1]
                - (idx_msk_neigh >= size_data[1]) * size_data[1]
            )

            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)
            input_list.append(input[id_msk_neigh])

        output[id_msk] = torch.median(torch.stack(input_list), dim=0).values
        mask[id_msk] = 0.0

    output = output.permute(2, 0, 1)
    mask = mask.permute(2, 0, 1)

    return output, mask


def generate_batch_masks(input):
    batch_size = input.shape[0]
    mask = []
    output = []
    for b in range(batch_size):
        o, m = generate_mask_base(input[b])
        mask.append(m)
        output.append(o)
    return torch.stack(output, dim=0), torch.stack(mask, dim=0)


def n2v2_backprob(model, noisy, optimizer, criterion):
    # noisy= noisy.permute(0, 2, 3, 1)
    # noisy = space_to_depth(noisy, 2)

    net_input, mask = generate_batch_masks(noisy)
    output = model(net_input, test=False)
    optimizer.zero_grad()
    loss = criterion(output * (1 - mask), noisy * (1 - mask))

    loss.backward()
    optimizer.step()
    return loss, output


def n2v2_evaluate(noisy_im, inferer, model):
    exp_output = inferer(noisy_im, model)
    return exp_output


# def generate_mask_median(input, ratio=0.9, size_window=(5, 5)):
#     input = input.permute(1, 2, 0)
#     size_data = input.shape
#     num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

#     mask = torch.ones(size_data).to("cuda")
#     output = input
#     half_size_row, half_size_col = size_window[0] // 2, size_window[1] // 2

#     for ich in range(size_data[2]):
#         idy_msk = torch.randint(0, size_data[0], (num_sample,), dtype=torch.long)
#         idx_msk = torch.randint(0, size_data[1], (num_sample,), dtype=torch.long)

#         id_msk = (idy_msk, idx_msk, ich)
#         median_list = []
#         for s in range(num_sample):
#             row, col, c = idy_msk[s], idx_msk[s], ich
#             neighborhood_rows = slice(max(0, row - half_size_row), min(input.shape[0], row + half_size_row+1))
#             neighborhood_cols = slice(max(0, col - half_size_col), min(input.shape[1], col + half_size_col+1))
#             neighborhood = input[neighborhood_rows, neighborhood_cols,c].flatten()
#             nei_mask = torch.arange(len(neighborhood)) != int(size_window[0]*size_window[1]/2)
#             median_list.append(torch.median(neighborhood[nei_mask]))

#         output[id_msk] = torch.stack(median_list)
#         mask[id_msk] = 0.0

#     output = output.permute(2, 0, 1)
#     mask = mask.permute(2, 0, 1)

#     return output, mask
