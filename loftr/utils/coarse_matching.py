import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e9


def mask_border(m, b: int, v):
    """Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


class CoarseMatching(nn.Module):
    def __init__(self, config, d_size):
        super().__init__()
        if config["match_type"] != "dual_softmax":
            raise NotImplementedError

        # general config
        self.thr = config["thr"]
        self.border_rm = config["border_rm"]

        # For matching
        self.temperature = config["dsmax_temperature"]
        self.d_size = d_size

        # -- # for training fine-level LoFTR
        self.train_coarse_percent = config["train_coarse_percent"]
        self.train_pad_num_gt_min = config["train_pad_num_gt_min"]

    def forward(self, feat_c0, feat_c1, hw0_i, hw1_i, hw0_c, hw1_c):
        """
        Args:
            feat_c0 (torch.Tensor): [N, L, C]
            feat_c1 (torch.Tensor): [N, S, C]
        Returns:
            mkpts0_c (torch.Tensor): [M, 2]
            mkpts1_c (torch.Tensor): [M, 2]
            conf_matrix (torch.Tensor): [M]
            b_ids (torch.Tensor): [M']
            i_ids (torch.Tensor): [M']
            j_ids (torch.Tensor): [M']
        NOTE: M' != M during training.
        """
        # normalize
        # feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])
        feat_c0, feat_c1 = map(
            lambda feat: feat / self.d_size**0.5, [feat_c0, feat_c1]
        )

        # sim_matrix_t = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        sim_matrix_orig = torch.matmul(feat_c0, feat_c1.permute((0, 2, 1)))
        sim_matrix = sim_matrix_orig / self.temperature
        # assert(torch.allclose(sim_matrix_t, sim_matrix, atol=1e-05))

        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        (
            gt_mask,
            b_ids,
            i_ids,
            j_ids,
            m_bids,
            mkpts0_c,
            mkpts1_c,
            mconf,
        ) = self.get_coarse_match(conf_matrix, hw0_i, hw1_i, hw0_c, hw1_c)
        return (
            conf_matrix,
            sim_matrix_orig,
            gt_mask,
            b_ids,
            i_ids,
            j_ids,
            m_bids,
            mkpts0_c,
            mkpts1_c,
            mconf,
        )

    @torch.no_grad()
    def get_coarse_match(
        self,
        conf_matrix: torch.Tensor,
        hw0_i: torch.Size,
        hw1_i: torch.Size,
        hw0_c: torch.Size,
        hw1_c: torch.Size,
    ):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            hw0_i (torch.Size): [H0, W0]
            hw1_i (torch.Size): [H1, W1]
            hw0_c (torch.Size): [H0, W0]
            hw1_c (torch.Size): [H1, W1]
        Returns:
            gt_mask (torch.Tensor): [M'],
            b_ids   (torch.Tensor): [M'],
            i_ids   (torch.Tensor): [M'],
            j_ids   (torch.Tensor): [M']
        """
        axes_lengths = {
            "h0c": hw0_c[0],
            "w0c": hw0_c[1],
            "h1c": hw1_c[0],
            "w1c": hw1_c[1],
        }

        # 1. confidence thresholding
        __device = conf_matrix.device
        mask = (conf_matrix > self.thr).to(dtype=torch.float32).to(__device)
        mask = torch.reshape(
            mask,
            (
                -1,
                axes_lengths["h0c"],
                axes_lengths["w0c"],
                axes_lengths["h1c"],
                axes_lengths["w1c"],
            ),
        )

        mask_border(mask, self.border_rm, 0)  # Replace False with 0
        mask = torch.reshape(
            mask,
            (
                -1,
                axes_lengths["h0c"] * axes_lengths["w0c"],
                axes_lengths["h1c"] * axes_lengths["w1c"],
            ),
        )

        # 2. mutual nearest
        mask = (
            mask
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        )

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        # if self.training:
        #     # NOTE:
        #     # The sampling is performed across all pairs in a batch without manually balancing
        #     # #samples for fine-level increases w.r.t. batch_size
        #     __device = conf_matrix.device

        #     num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
        #     num_matches_train = int(num_candidates_max * self.train_coarse_percent)
        #     num_matches_pred = len(b_ids)
        #     assert (
        #         self.train_pad_num_gt_min < num_matches_train
        #     ), "min-num-gt-pad should be less than num-train-matches"

        #     # pred_indices is to select from prediction
        #     if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
        #         pred_indices = torch.arange(num_matches_pred, device=__device)
        #     else:
        #         pred_indices = torch.randint(
        #             num_matches_pred,
        #             (num_matches_train - self.train_pad_num_gt_min,),
        #             device=__device,
        #         )

        #     # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
        #     gt_pad_indices = torch.randint(
        #         len(data["spv_b_ids"]),
        #         (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
        #         device=__device,
        #     )
        #     mconf_gt = torch.zeros(
        #         len(data["spv_b_ids"]), device=__device
        #     )  # set conf of gt paddings to all zero

        #     b_ids, i_ids, j_ids, mconf = map(
        #         lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
        #         *zip(
        #             [b_ids, data["spv_b_ids"]],
        #             [i_ids, data["spv_i_ids"]],
        #             [j_ids, data["spv_j_ids"]],
        #             [mconf, mconf_gt],
        #         )
        #     )

        # 4. Update with matches in original image resolution
        scale = hw0_i[0] / hw0_c[0]
        scale0 = scale
        scale1 = scale
        mkpts0_c = (
            torch.stack(
                [i_ids % hw0_c[1], torch.div(i_ids, hw0_c[1], rounding_mode="floor")],
                dim=1,
            )
            * scale0
        )
        mkpts1_c = (
            torch.stack(
                [j_ids % hw1_c[1], torch.div(j_ids, hw1_c[1], rounding_mode="floor")],
                dim=1,
            )
            * scale1
        )

        gt_mask = mconf == 0
        mask = mconf != 0

        return (
            gt_mask,
            b_ids,
            i_ids,
            j_ids,
            b_ids[mask],
            mkpts0_c[mask],
            mkpts1_c[mask],
            mconf[mask],
        )
