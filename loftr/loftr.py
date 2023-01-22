import torch
import torch.nn as nn

from .backbone import build_backbone
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.position_encoding import PositionEncodingSine
from .utils.fine_matching import FineMatching


class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config["coarse"]["d_model"], temp_bug_fix=config["coarse"]["temp_bug_fix"]
        )
        self.loftr_coarse = LocalFeatureTransformer(config["coarse"])
        self.coarse_matching = CoarseMatching(
            config["match_coarse"], config["coarse"]["d_model"]
        )
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_preprocess = FinePreprocess(config)
        self.fine_matching = FineMatching()

    def backbone_forward(self, img0, img1):
        """
        'img0': (torch.Tensor): (N, 1, H, W)
        'img1': (torch.Tensor): (N, 1, H, W)
        """

        # we assume that data['hw0_i'] == data['hw1_i'] - faster & better BN convergence
        feats_c, feats_f = self.backbone(torch.cat([img0, img1], dim=0))  # type: ignore

        (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(
            img0.shape[0]
        ), feats_f.split(img0.shape[0])

        return feat_c0, feat_f0, feat_c1, feat_f1

    def forward(self, img0, img1):
        """
        'img0': (torch.Tensor): (N, 1, H, W)
        'img1': (torch.Tensor): (N, 1, H, W)
        """
        # 1. Local Feature CNN
        hw0_i = img0.shape[2:]
        hw1_i = img1.shape[2:]

        feat_c0, feat_f0, feat_c1, feat_f1 = self.backbone_forward(img0, img1)
        hw0_c = feat_c0.shape[2:]
        hw1_c = feat_c1.shape[2:]
        hw0_f = feat_f0.shape[2:]
        # hw1_f = feat_f1.shape[2:]

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        # feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        feat_c0 = torch.flatten(self.pos_encoding(feat_c0), 2, 3).permute(0, 2, 1)
        feat_c1 = torch.flatten(self.pos_encoding(feat_c1), 2, 3).permute(0, 2, 1)

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1)

        # 3. match coarse-level
        (
            conf_matrix,
            sim_matrix,
            gt_mask,
            b_ids,
            i_ids,
            j_ids,
            m_bids,
            mkpts0_c,
            mkpts1_c,
            mconf,
        ) = self.coarse_matching.forward(feat_c0, feat_c1, hw0_i, hw1_i, hw0_c, hw1_c)

        # 4. fine-level loftr module
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess.forward(
            feat_f0, feat_f1, feat_c0, feat_c1, hw0_f, hw0_c, b_ids, i_ids, j_ids
        )
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(
                feat_f0_unfold, feat_f1_unfold
            )

        # 5. Match fine-level
        expec_f, mkptsf_0, mkptsf_1 = self.fine_matching.forward(
            feat_f0_unfold, feat_f1_unfold, mconf, mkpts0_c, mkpts1_c, hw0_i, hw1_i
        )

        return mkptsf_0, mkptsf_1, conf_matrix  # , sim_matrix, mkptsf_0, mkptsf_1

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
