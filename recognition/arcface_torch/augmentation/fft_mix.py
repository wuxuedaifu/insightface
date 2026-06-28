import torch


def amplitude_spectrum_mix(src: torch.Tensor, ref: torch.Tensor,
                           ratio: float = 0.1) -> torch.Tensor:
    """Blend low-frequency amplitude of ref into src, preserving src's phase.

    Implements the FFT augmentation from TransFace (ICCV 2023):
    2D FFT both images, blend amplitude in the central ratio×ratio region
    of the shifted spectrum, inverse FFT to get the augmented image.

    Args:
        src: float tensor (B, C, H, W), pixel values in [0, 1]
        ref: float tensor (B, C, H, W), randomly sampled reference batch
        ratio: fraction of the spectrum's spatial extent to blend (0–1)
    Returns:
        Augmented float tensor with same shape as src, values clamped to [0, 1]
    """
    B, C, H, W = src.shape

    F_src = torch.fft.fft2(src)
    F_ref = torch.fft.fft2(ref)

    # Shift DC component to center
    F_src_s = torch.fft.fftshift(F_src)
    F_ref_s = torch.fft.fftshift(F_ref)

    amp_src = F_src_s.abs()
    phase_src = F_src_s.angle()
    amp_ref = F_ref_s.abs()

    # Central blend region
    h_crop = int(H * ratio)
    w_crop = int(W * ratio)
    h0 = H // 2 - h_crop // 2
    h1 = h0 + h_crop
    w0 = W // 2 - w_crop // 2
    w1 = w0 + w_crop

    amp_mixed = amp_src.clone()
    amp_mixed[:, :, h0:h1, w0:w1] = (
        0.5 * amp_src[:, :, h0:h1, w0:w1]
        + 0.5 * amp_ref[:, :, h0:h1, w0:w1]
    )

    F_out = amp_mixed * torch.exp(1j * phase_src)
    F_out = torch.fft.ifftshift(F_out)
    out = torch.fft.ifft2(F_out).real
    return out.clamp(0.0, 1.0)
