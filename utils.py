import cv2
import numpy as np


def get_coarse_match(conf_matrix, input_height, input_width, resolution):
    """
        Predicts coarse matches from conf_matrix
    Args:
        resolution: image
        input_width:
        input_height:
        conf_matrix: [N, L, S]

    Returns:
        mkpts0_c: [M, 2]
        mkpts1_c: [M, 2]
        mconf: [M]
    """

    hw0_i = (input_height, input_width)
    hw0_c = (input_height // resolution, input_width // resolution)
    hw1_c = hw0_c  # input images have the same resolution
    feature_num = hw0_c[0] * hw0_c[1]

    # 3. find all valid coarse matches
    # this only works when at most one `True` in each row
    mask = conf_matrix
    all_j_ids = mask.argmax(axis=2)
    j_ids = all_j_ids.squeeze(0)
    b_ids = np.zeros_like(j_ids, dtype=np.long)
    i_ids = np.arange(feature_num, dtype=np.long)

    mconf = conf_matrix[b_ids, i_ids, j_ids]

    # 4. Update with matches in original image resolution
    scale = hw0_i[0] / hw0_c[0]
    mkpts0_c = np.stack(
        [i_ids % hw0_c[1], np.trunc(i_ids / hw0_c[1])],
        axis=1) * scale
    mkpts1_c = np.stack(
        [j_ids % hw1_c[1], np.trunc(j_ids / hw1_c[1])],
        axis=1) * scale

    return mkpts0_c, mkpts1_c, mconf


def make_query_image(frame, img_size):
    # ratio preserving resize
    img_h, img_w, _ = frame.shape
    scale_h = img_size[1] / img_h
    scale_w = img_size[0] / img_w
    scale_max = max(scale_h, scale_w)
    new_size = (int(img_w * scale_max), int(img_h * scale_max))
    query_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    query_img = cv2.resize(query_img, new_size, interpolation=cv2.INTER_LINEAR)
    # center crop
    x = new_size[0] // 2 - img_size[0] // 2
    y = new_size[1] // 2 - img_size[1] // 2
    query_img = query_img[y:y + img_size[1], x:x + img_size[0]]
    return query_img
