# flops_ae_flow_fix.py
import argparse
import torch
from model import ae_flow

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fmt_m(n):
    return f"{n/1e6:.2f}M"

def flops_params_thop(model, x):
    from thop import profile
    macs, params = profile(model, inputs=(x,), verbose=False)
    flops_g = float(2 * macs) / 1e9  # FLOPs ≈ 2*MACs
    return flops_g, int(params)

class RecOnly(torch.nn.Module):
    """Wrapper: only computes rec_img = AE_FLOW(img)[0]."""
    def __init__(self, ae_flow_model):
        super().__init__()
        self.m = ae_flow_model

    def forward(self, x):
        out = self.m(x)
        # training uses: rec_img, z_hat, jac = model(img)
        rec = out[0] if isinstance(out, (tuple, list)) else out
        # make scalar output so thop won't complain about tuples
        return rec.sum()

@torch.no_grad()
def main(subnet="conv_type", in_channels=2, image_size=256, device="cuda:0"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ae_flow.AE_FLOW(subnet=subnet).to(device).eval()
    x = torch.zeros(1, in_channels, image_size, image_size, device=device)

    full_params = count_parameters(model)

    # 1) try FLOPs for reconstruction path only
    rec_wrap = RecOnly(model).to(device).eval()
    rec_flops_g = 0.0
    rec_params = count_parameters(model)  # same params, but FLOPs only reflects rec forward ops
    rec_ok = True
    try:
        rec_flops_g, _ = flops_params_thop(rec_wrap, x)
    except Exception as e:
        rec_ok = False
        err = repr(e)

    # Print in your requested style
    if rec_ok:
        print(f"FLOPs (Forward): {rec_flops_g:.2f}G")
        print(f"  FLOPs (Full)   : {rec_flops_g:.2f}G")
        print(f"  Parameters     : {fmt_m(full_params)}")
        print(f"  FLOPs          : {rec_flops_g:.2f}G")
    else:
        # give explicit reason instead of silent 0
        print("FLOPs (Forward): 0.00G")
        print("  FLOPs (Full)   : 0.00G")
        print(f"  Parameters     : {fmt_m(full_params)}")
        print("  FLOPs          : 0.00G")
        print("\n[Hint] thop could not profile this model forward. Error:")
        print(err)
        print("\nTry: profile only the encoder/decoder submodules (conv parts) if AE_FLOW contains custom flow ops.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subnet", type=str, default="conv_type")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args.subnet, args.in_channels, args.image_size, args.device)