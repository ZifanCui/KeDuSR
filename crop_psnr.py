import cv2
import numpy as np


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def calculate_psnr(img, img2, crop_border=None, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if not crop_border:
        mse = np.mean((img - img2)**2)
        if mse == 0:
            return float('inf')
        return 10. * np.log10(255. * 255. / mse)
    
    else:
        H, W, C = img.shape
        mse_mat = (img - img2)**2
        mes_mean = np.mean(mse_mat)  #258.4610880118649


        mes_center = mse_mat[crop_border[0]: crop_border[1], crop_border[2]: crop_border[3], ...]

        center_area = (crop_border[1]-crop_border[0]) * (crop_border[3] - crop_border[2]) * 3
        mes_center_mean = mes_center.sum()/center_area  #126.91002338435374

        corner_area = H*W*3 - center_area
        mse_mat[crop_border[0]: crop_border[1], crop_border[2]: crop_border[3], ...] = 0
        mes_corner_mean = mse_mat.sum()/corner_area  #267.2311589870323

        # avg = (mes_corner_mean*3 + mes_center_mean)/4


        if mes_mean == 0 or mes_center_mean==0 or mes_corner_mean==0:
            return float('inf')
        return [10. * np.log10(255. * 255. / mes_mean), 
                10. * np.log10(255. * 255. / mes_center_mean), 10. * np.log10(255. * 255. / mes_corner_mean)]


def calculate_ssim(img, img2, crop_border=None, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)


    if not crop_border:
        ssims = []
        for i in range(img.shape[2]):
            ssims.append(_ssim(img[..., i], img2[..., i]))
        return np.array(ssims).mean()
    
    else:
        ssims_all = []
        ssims_center = []
        ssims_corner = []
        for i in range(img.shape[2]):
            

            img_ = img[..., i]
            img2_ = img2[..., i]
            c1 = (0.01 * 255)**2
            c2 = (0.03 * 255)**2
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img_, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
            mu2 = cv2.filter2D(img2_, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img_**2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2_**2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img_ * img2_, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

            H, W= ssim_map.shape

            all = ssim_map.mean() 

            center_area = (crop_border[1]-crop_border[0]) * (crop_border[3] - crop_border[2]) 

            center = ssim_map[crop_border[0]: crop_border[1], crop_border[2]: crop_border[3]].mean()

            corner_area = H*W - center_area

            ssim_map[crop_border[0]: crop_border[1], crop_border[2]: crop_border[3]] = 0

            corner = ssim_map.sum() / corner_area

            ssims_all.append(all)
            ssims_center.append(center)
            ssims_corner.append(corner)

        return [np.array(ssims_all).mean(), np.array(ssims_center).mean(), np.array(ssims_corner).mean()]


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()



if __name__ == '__main__':
    pass