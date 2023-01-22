import torch
import torch.nn as nn
import torch.nn.functional as F


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config["fine_concat_coarse_feat"]
        self.W = self.config["fine_window_size"]

        d_model_c = self.config["coarse"]["d_model"]
        d_model_f = self.config["fine"]["d_model"]
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(
        self, feat_f0, feat_f1, feat_c0, feat_c1, hw0_f, hw0_c, b_ids, i_ids, j_ids
    ):
        W = self.W
        stride = torch.div(hw0_f[0], hw0_c[0], rounding_mode="floor").repeat(2)

        if b_ids.shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(
            feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2  # type: ignore
        )
        feat_f0_unfold = torch.reshape(
            feat_f0_unfold,
            (
                feat_f0_unfold.shape[0],
                -1,
                W**2,
                feat_f0_unfold.shape[2],
            ),
        ).permute(0, 3, 2, 1)

        feat_f1_unfold = F.unfold(
            feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2  # type: ignore
        )
        feat_f1_unfold = torch.reshape(
            feat_f1_unfold,
            (
                feat_f1_unfold.shape[0],
                -1,
                W**2,
                feat_f1_unfold.shape[2],
            ),
        ).permute(0, 3, 2, 1)

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[b_ids, i_ids]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[b_ids, j_ids]

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(
                torch.cat(
                    [
                        feat_c0[b_ids, i_ids],
                        feat_c1[b_ids, j_ids],
                    ],
                    0,
                )
            )  # [2n, c]

            feat_cf_win = self.merge_feat(
                torch.cat(
                    [
                        torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                        feat_c_win.unsqueeze(1).repeat(1, W**2, 1),  # [2n, ww, cf]
                    ],
                    -1,
                )
            )
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold
