def compute_p_and_r(pred, target):
    # compute the p and r of the foreground
    # batchsize = 1
    TP = np.sum(target * pred)
    p = TP / np.sum(pred)
    r = TP / np.sum(target)
    # print('TP, p ,r:',TP, p ,r)
        
    return p, r

def compute_PWC(pred, target):
    # first: compute TP_TN_FP_FN
    TP = np.sum(target * pred)
    FP = np.sum(pred) - TP
    FN = np.sum(target) - TP
    # TN = target.shape[0] * target.shape[1] - TP - FP - FN

    PWC = 100 * (FN + FP) / (target.shape[0] * target.shape[1])

    # print('TP, p ,r:',TP, p ,r)
    
    return PWC


def compute_F_score_and_PWC_of_middle_fuse(pred, target):
    # F_score_list = []
    # thresh_list = []
    # print('max, min:', pred.max(), pred.min())
    # print('np.sum(target):', np.sum(target))

    if np.sum(target) != 0:
        thresh = 230
        # print('thresh:', thresh)
        pred_ = np.where(pred > np.uint8(thresh), np.uint8(1), np.uint8(0))
        p_, r_ = compute_p_and_r(pred_, target)
        PWC = compute_PWC(pred_, target)
        if (p_ == 0) or (r_ == 0):
            return None, None, None, None, None
        F_score = 2 * p_ * r_ / (p_ + r_)
        # print('F_score_thresh,p,r,F_score:',thresh, p_, r_, F_score)

        return F_score, thresh, p_, r_, PWC
    else:
        return None, None, None, None, None

def compute_F_score_and_PWC_of_final(pred, target):
    # F_score_list = []
    # thresh_list = []
    # print('max, min:', pred.max(), pred.min())
    # print('np.sum(target):', np.sum(target))

    if np.sum(target) != 0:
        thresh = 230 # general:255, nightvideos:243
        # print('thresh:', thresh)
        pred_ = np.where(pred > np.uint8(thresh), np.uint8(1), np.uint8(0))
        p_, r_ = compute_p_and_r(pred_, target)
        PWC = compute_PWC(pred_, target)
        if (p_ == 0) or (r_ == 0):
            return None, None, None, None, None
        F_score = 2 * p_ * r_ / (p_ + r_)
        # print('F_score_thresh,p,r,F_score:',thresh, p_, r_, F_score)


        return F_score, thresh, p_, r_, PWC
    else:
        return None, None, None, None, None
